import torch as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import glob
import random
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from early_stopping import EarlyStopping
from plotting import plot_training_validation_losses
from helper_functions import zip_files, get_datetime, log_to_tensorboard, get_dfs_from_clims, create_path

class CustomDataset(Dataset):
    '''Class that extends 'Dataset' to seperate the variables of our dataframes
    
    Attributes
    ----------
    y_nrr : torch.Tensor
        target variable of our data
    x : torch.Tensor
        features of our data
    rest : torch.Tensor
        features that identify each simulation (factorials of data generation)  
    n_samples : int
        total samples of our dataset  
    '''
    
    def __init__(self, data_df):
        self.y_nrr = T.tensor(data_df.iloc[:, 2].values)
        self.x = T.tensor(data_df.iloc[:, 3:-4].values)      
        self.rest = T.tensor(data_df[['SoilFertility', 'Irrigation', 'FertMonth', 'FertDay', 'FertRate_orig', 'SoilWater_orig', 'Year']].values)
        
        self.n_samples = data_df.shape[0]
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x[index], self.y_nrr[index], self.rest[index]

class NRRPredictor(T.nn.Module):
    '''Module that contains the encoder and regression head after removing the decoder
    
    Attributes
    ----------
    self.encoder : Encoder
        Encoder previously part of the autoencoder
    self.nrr_regressor : NRRRegressor
        Regression head
    '''
    
    #'encoder' trained instance of Encoder
    #'latent_dims' (int) is size of bottleneck
    def __init__(self, encoder, latent_dims):
        super().__init__()
        self.encoder = encoder #we don't need to create one, we reuse the one from autoencoder
        self.nrr_regressor = NRRRegressor(latent_dims)

    def forward(self, x):
        x, rescon_l1, rescon_l2 = self.encoder(x)
        x = self.nrr_regressor(x)
        return x

class NRRRegressor(T.nn.Module):
    '''Module that implements a MLP
    
    Attributes
    ----------
    self.linear1 : torch.nn.modules.linear.Linear
        Input layer
    self.linear2 : torch.nn.modules.linear.Linear
        Hidden layer
    self.linear3 : torch.nn.modules.linear.Linear
        Output layer
    '''
    
    def __init__(self, latent_dims):
        super().__init__()
        self.linear1 = T.nn.Linear(latent_dims, int(1.5*latent_dims))
        self.linear2 = T.nn.Linear(int(1.5*latent_dims), int(1.5*latent_dims))
        self.linear3 = T.nn.Linear(int(1.5*latent_dims), 1)

        #weight and bias initialization
        T.nn.init.kaiming_uniform_(self.linear1.weight)
        T.nn.init.ones_(self.linear1.bias)
        T.nn.init.kaiming_uniform_(self.linear2.weight)
        T.nn.init.ones_(self.linear2.bias)
        T.nn.init.kaiming_uniform_(self.linear3.weight)
        T.nn.init.ones_(self.linear3.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        #x = F.dropout(x, 0.2)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        #x = F.dropout(x, 0.2)
        x = self.linear3(x)
        return x

class Encoder(T.nn.Module):
    '''Module that implements the encoder of the autoencoder
    
    Attributes
    ----------
    linear1 : torch.nn.modules.linear.Linear
        Input layer
    linear2 : torch.nn.modules.linear.Linear
        Hidden layer
    linear3 : torch.nn.modules.linear.Linear
        Output layer, outputs the bottleneck
    '''
    
    def __init__(self, latent_dims):
        super().__init__()
        self.linear1 = T.nn.Linear(425, 300)
        self.linear2 = T.nn.Linear(300, 200)
        self.linear3 = T.nn.Linear(200, latent_dims)

        #weight and bias initialization
        T.nn.init.kaiming_uniform_(self.linear1.weight)
        T.nn.init.ones_(self.linear1.bias)
        T.nn.init.kaiming_uniform_(self.linear2.weight)
        T.nn.init.ones_(self.linear2.bias)
        T.nn.init.kaiming_uniform_(self.linear3.weight)
        T.nn.init.ones_(self.linear3.bias)

    def forward(self, x):
        x = self.linear1(x)
        
        rescon_l1 = x
        
        x = F.leaky_relu(x)
        x = F.dropout(x, 0.1)
        x = self.linear2(x)
        
        rescon_l2 = x
        
        x = F.leaky_relu(x)
        x = F.dropout(x, 0.1)
        x = self.linear3(x)                
        x = F.leaky_relu(x)
        
        return [x, rescon_l1, rescon_l2]

class Decoder(T.nn.Module):
    '''Module that implements the decoder of the autoencoder
    
    Attributes
    ----------
    linear1 : torch.nn.modules.linear.Linear
        Input layer, takes the bottleneck of the encoder
    linear2 : torch.nn.modules.linear.Linear
        Hidden layer
    linear3 : torch.nn.modules.linear.Linear
        Output layer, outputs the bottleneck
    '''
    
    def __init__(self, latent_dims):
        super().__init__()
        self.linear1 = T.nn.Linear(latent_dims, 200)
        self.linear2 = T.nn.Linear(200, 300)
        self.linear3 = T.nn.Linear(300, 425)

        #weight and bias initialization
        T.nn.init.kaiming_uniform_(self.linear1.weight)
        T.nn.init.ones_(self.linear1.bias)
        T.nn.init.kaiming_uniform_(self.linear2.weight)
        T.nn.init.ones_(self.linear2.bias)
        T.nn.init.kaiming_uniform_(self.linear3.weight)
        T.nn.init.ones_(self.linear3.bias)

    def forward(self, x, rescon_l1, rescon_l2):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        
        x = x + rescon_l2
        x = F.leaky_relu(x)
        
        x = F.dropout(x, 0.1)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        
        x = x + rescon_l1
        x = F.leaky_relu(x)
        
        x = F.dropout(x, 0.1)
        x = self.linear3(x)
        return x

class Autoencoder(T.nn.Module):
    '''Module that brings together the encoder and decoder
    
    Attributes
    ----------
    encoder : Encoder
        encoder module
    decoder : Decoder
        decoder module
    '''
    
    #'latent_dims' (int) is size of bottleneck
    def __init__(self, latent_dims):
        super().__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        enc_out, rescon_l1, rescon_l2 = self.encoder(x)
        dec_out = self.decoder(enc_out, rescon_l1, rescon_l2)
        return dec_out
    
def validate(net, task, dataloader, loss):
    '''Returns  list
    
    net: DualHeadAutoencoder, the network under validation
    task: str, shows if we pretraing the encoder or not ('train_encoder' means pretrain)
    dataloader: torch.utils.data.dataloader.DataLoader, the dataloader of the validation set
    loss: torch.nn.MSELoss, loss function

    Calculates the validation loss of each batch of an epoch and returns them 
    '''
    
    validation_losses = []
        
    with T.no_grad():    
            net.train(mode=False)
            for i, (x, y_nrr, rest) in enumerate(dataloader):
                
                x = x.float()               
                #x = x.to(device)
                y_nrr = y_nrr.float()               
                #y = y.to(device)
                
                if task=='train_encoder':
                    dec_out = net(x)
                    l = loss(dec_out, x)
                else:
                    output = net(x)
                    l = loss(output, y_nrr.unsqueeze(1))

                validation_losses.append(l.item())
                            
    return validation_losses

def train(net, task, train_dataloader, validation_dataloader, num_epochs, early_stopping_patience, tensorboard_writer):
    '''Returns a tuple containing the trained network, average training loss per epoch, average validation loss per epoch
    
    net: DualHeadAutoencoder, the network to be trained
    task: str, shows if we pretraing the encoder or not ('train_encoder' means pretrain)
    train_dataloader: torch.utils.data.dataloader.DataLoader, the dataloader of the training set
    validation_dataloader: torch.utils.data.dataloader.DataLoader, the dataloader of the validation set
    num_epochs: int, how many epochs to train
    early_stopping_patience: int, how many epochs to continue with consecutive validation losses above the lowest validation loss
    tensorboard_writer: torch.utils.tensorboard.writer.SummaryWriter obejct, for logging purposes

    Trains a neural net
    '''
    
    #helper operations
    early_stopper = EarlyStopping(patience=early_stopping_patience, verbose=True)
    
    avg_train_losses = []
    avg_valid_losses = []
    
    #optimizer
    optimizer = T.optim.AdamW(net.parameters(), lr=0.0003)
    loss = T.nn.MSELoss()
    
    
    if task != 'train_encoder':
        print('Encoder will not be trained')

        net.encoder.requires_grad_(False)
        
        optimizer = T.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0003)      
    else:
        print('Encoder will be trained')
    
    for epoch in range(num_epochs):
    
        net.train(mode=True)
        training_losses=[]
        
        for i, (x, y_nrr, rest) in enumerate(train_dataloader):
            
            x = x.float()       
            #x = x.to(device)
            y_nrr = y_nrr.float()       
            #y = y.to(device) for some reason that's very expensive
            
            
            if task=='train_encoder':
                dec_out = net(x)
                l = loss(dec_out, x)
            else:
                output = net(x)
                l = loss(output, y_nrr.unsqueeze(1))
            net.zero_grad() #clears all parameters of the model while optimizer.zero_grad() clears the gradients of the parameters passed to it
            l.backward()
            optimizer.step()
            
            training_losses.append(l.item())
            
        validation_losses = validate(net, task, validation_dataloader, loss)        
                        
        train_loss = np.average(training_losses)
        valid_loss = np.average(validation_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        log_to_tensorboard(tensorboard_writer, net, epoch, train_loss, valid_loss)
        
        early_stopper(valid_loss, net, epoch, num_epochs)
        if early_stopper.early_stop == True:
            print('Early stopping triggered')
            break
                
    tensorboard_writer.flush()
    
    return net, avg_train_losses, avg_valid_losses
            
def test(net, dataloader):
    '''Returns a tuple containing the regression rmse and a dataframe with the test data including the prediction for each sample
    
    net: NRRPredictor, the network to be trained
    dataloader: torch.utils.data.dataloader.DataLoader, a training or test dataloader

    Evaluates the provided net
    '''
    
    df = pd.DataFrame(columns=['SoilFertility', 'Irrigation', 'FertMonth', 'FertDay', 'FertRate_orig', 'SoilWater_orig', 'Year', 'target_var', 'NRR', 'Residual'])
    results_tensor = T.tensor([])
        
    with T.no_grad():
        
        net.train(mode=False)
        
        errors=np.array([])
        
        for i, (x, y_nrr, rest) in enumerate(dataloader):

            if i>0 and i%500 == 0:
                print('Testing : iteration ' + str(i))
                                
            x = x.float()            
            #x = x.to(device)
            y_nrr = y_nrr.float()       
            #y = y.to(device)
            
            prediction = net(x)
            
            squared_error = (prediction-y_nrr.unsqueeze(1))**2
            errors = np.append(errors, squared_error.detach())            
            
            results_tensor = T.cat((results_tensor, 
                                    T.cat((rest, 
                                            y_nrr.unsqueeze(1),
                                            prediction,
                                            T.abs(y_nrr.unsqueeze(1)-prediction)), dim=1)), dim = 0)
            results_tensor=results_tensor.detach()
            
        error = math.sqrt(errors.mean()) 
        
    
        return [error, df.append(pd.DataFrame(results_tensor, columns=df.columns).astype('float'), ignore_index=True)]

def run(scenario_id, batch_size, latent_dims, autoencoder_epochs, regressor_epochs, patience, train_clims, validation_clims, test_clim, metamodel_type, test_type, scenario_folder, filename_train, filename_validate, filename_test, irrigation, base_path, save_path):
    '''Returns None
    
    scenario_id: str, training scenario identifier
    batch_size: int, batch size for training validation test
    latent_dims: int, size of the bottleneck
    autoencoder_epochs: int, how many epochs the autoencoder will be trained
    regressor_epochs: int, how many epochs the regression head will be trained after removing the decoder
    patience: int, how many epochs to continue with consecutive validation losses above the lowest validation loss
    train_clims: list, in which climates to train
    validation_clims: list, in which climates to validate
    test_clim: str, in which climate to test
    metamodel_type: str, 'local' 'regional' 'national'
    test_type: str, 'known' 'unknown'
    scenario_folder: str, in which scenario folder to store the results
    filename_train: str, name of the file containing the training data for each climate (assumes that in all climates the filename is the same)
    filename_validate: str, name of the file containing the validation data for each climate (assumes that in all climates the filename is the same)
    filename_test: str, name of the file containing the test climate data
    irrigation: str, shows which data to keep, either 'irrigated', 'non_irrigated', or else keeps both
    base_path: str, root of the data and result folders
    save_path: str, path to store the results
    
    Orchestrates the training and evaluation procedures
    '''    
    
    training_df, validation_df, test_df = get_dfs_from_clims(base_path, train_clims, validation_clims, test_clim, filename_train, filename_validate, filename_test, irrigation)
    
    #create Dataset objects
    print('Reading datasets')
    train_dataset = CustomDataset(training_df)
    validation_dataset = CustomDataset(validation_df)
    test_dataset =  CustomDataset(test_df)
    print('Datasets read')
    
    #create Dataloader objects
    train_dataloader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers = True)
    validation_dataloader = DataLoader(dataset = validation_dataset, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers = True)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers = True) 
    
    #initialize tensorboard writer
    tensorboard_writer = SummaryWriter(comment = scenario_id)
    
    
    #initialize the net
    net = Autoencoder(latent_dims)
    net.to(device)
    
    #train the net
    auto_net, avg_train_losses, avg_valid_losses = train(net, 'train_encoder', train_dataloader, validation_dataloader, autoencoder_epochs, patience, tensorboard_writer)
    T.save(auto_net.encoder.state_dict(), 'saved_encoder.pt')
    
    checkpointed_net = Autoencoder(latent_dims)
    checkpointed_net.load_state_dict(T.load('model_checkpoint.pt'))
    
    #freeze encoder, train MLP
    net = NRRPredictor(auto_net.encoder, latent_dims)
    net, avg_train_losses, avg_valid_losses = train(net, 'nrr', train_dataloader, validation_dataloader, regressor_epochs, patience, tensorboard_writer)
    
    checkpointed_net = NRRPredictor(auto_net.encoder, latent_dims)
    checkpointed_net.load_state_dict(T.load('model_checkpoint.pt'))
    
    #test the fully trained net
    training_error, trraining_predictions_df = test(checkpointed_net, train_dataloader)
    s_f_tr = scenario_id + ', ' + 'Fully trained autoencoder net train rmse:' + str(training_error)
    print(s_f_tr)
    
    testing_error, testing_predictions_df  = test(checkpointed_net, test_dataloader)
    s_f_te = scenario_id + ', ' + 'Fully trained autoencoder net test rmse:' + str(testing_error)
    print(s_f_te)
    
    #add constant columns which will allow to compare to random forest
    testing_predictions_df['Metamodel type'] = metamodel_type
    testing_predictions_df['Test type'] = test_type
    testing_predictions_df['Location'] = test_clim
    
    create_path(save_path)
    testing_predictions_df.to_csv(save_path + '/predictions_testing_' + scenario_id + '_' + test_clim + '.csv', index=False)
       
    plot_training_validation_losses('autoencoder_' + scenario_id, autoencoder_epochs, avg_train_losses, avg_valid_losses)
    zip_files(scenario_id, glob.glob('*.txt') + glob.glob('*.png') + glob.glob('*.py') + glob.glob('*.pt'))



if __name__ == '__main__':
    device =  T.device('cpu')
    T.use_deterministic_algorithms(True)
    
    print('Started at' , get_datetime())

    base_path = '...'

    for seed in [1, 32, 86, 1001, 40000]:
    
        print('Doing seed', seed)
               
        T.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        for train_clims, validation_clims, test_clim, metamodel_type, test_type, \
            scenario_folder, \
            filename_train, \
            filename_validate, \
            filename_test, \
            irrigation in [[['Clim1'], ['Clim1'], 'Clim1', 'local', 'known', \
                            'one_hot', \
                            'train_standardized_fert_soilwater.csv', \
                            'validation_standardized_fert_soilwater.csv', \
                            'test_standardized_fert_soilwater.csv', \
                            'both'],
                            [['Clim5'], ['Clim5'], 'Clim5', 'local', 'known', \
                            'one_hot', \
                            'train_standardized_fert_soilwater.csv', \
                            'validation_standardized_fert_soilwater.csv', \
                            'test_standardized_fert_soilwater.csv', \
                            'both']]:
                
                print('Testing in ', test_clim)
                print('Irrigation', irrigation)

                for latent_dims in [120]:
                    run(scenario_id = 'seed_' + str(seed),
                        batch_size = 64, 
                        latent_dims = latent_dims,
                        autoencoder_epochs = 60,
                        regressor_epochs = 60,
                        patience = 300,
                        train_clims = train_clims,
                        validation_clims = validation_clims,
                        test_clim = test_clim,
                        metamodel_type = metamodel_type,
                        test_type = test_type,
                        scenario_folder = scenario_folder,
                        filename_train = filename_train,
                        filename_validate = filename_validate, 
                        filename_test = filename_test,
                        irrigation = irrigation,
                        base_path = base_path,
                        save_path = base_path + 'predictions/' + test_clim + '/' + scenario_folder + '/res_autoencoder_transfer_adamw_fert_soilwater/' + test_type + '/' + metamodel_type + '/')

    print('Ended at ' , get_datetime())   




