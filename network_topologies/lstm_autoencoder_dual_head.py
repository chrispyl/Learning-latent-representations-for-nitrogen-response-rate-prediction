import torch as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import glob
import sys
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from early_stopping import EarlyStopping
from helper_functions import get_datetime, log_to_tensorboard, get_dfs_from_clims, create_path
from plotting import plot_training_validation_losses

class CustomDataset(Dataset):
    '''Class that extends 'Dataset' to seperate the variables of our dataframes
    
    Attributes
    ----------
    y_nrr : torch.Tensor
        target variable of our data
    x_timeseries : torch.Tensor
        timeseries features of our data
    x_scalars : torch.Tensor
        scalar features of our data
    rest : torch.Tensor
        features that identify each simulation (factorials of data generation)  
    timesteps : int
        number of steps in timeseries features  
    n_samples : int
        total samples of our dataset  
    '''
    
    def __init__(self, data_df, timesteps):
        self.y_nrr = T.tensor(data_df.iloc[:, 2].values)
        self.x_timeseries = T.tensor(data_df.iloc[:, 3:-10].values)
        self.x_scalars = T.tensor(data_df.iloc[:, -10:-5].values)          
        self.rest = T.tensor(data_df[['SoilFertility', 'Irrigation', 'FertMonth', 'FertDay', 'FertRate_orig', 'SoilWater_orig', 'Year']].values) #used to track which simulation each sample referes to
        
        self.timesteps = timesteps
        self.n_samples = data_df.shape[0]
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return T.reshape(self.x_timeseries[index], (self.timesteps, self.x_timeseries[index].shape[0]//self.timesteps)), self.x_scalars[index], self.y_nrr[index], self.rest[index] #reshapes time series features to fit the lstm setup

class NRRRegressor(T.nn.Module):
    '''Module that implements the regression head of the dual-head autoencoder
    
    Attributes
    ----------
    self.linear1 : torch.nn.modules.linear.Linear
        Input layer
    self.linear2 : torch.nn.modules.linear.Linear
        Hidden layer
    self.linear3 : torch.nn.modules.linear.Linear
        Output layer
    '''
    
    #'bottleneck_features' (int) is the number of features on the output of the encoder
    #'timesteps' (int) is the number of days in each timeseries
    #'scalars_n' (int) is the total number of scalars
    def __init__(self, bottleneck_features, timesteps, scalars_n):
        super().__init__()
        total_inputs = bottleneck_features * timesteps + scalars_n #timeseries and scalars will go together as input to the input layer
        self.linear1 = T.nn.Linear(total_inputs, total_inputs//2)
        self.linear2 = T.nn.Linear(total_inputs//2, total_inputs//4)
        self.linear3 = T.nn.Linear(total_inputs//4, 1)

        #weight and bias initialization
        T.nn.init.kaiming_uniform_(self.linear1.weight)
        T.nn.init.kaiming_uniform_(self.linear2.weight)
        T.nn.init.kaiming_uniform_(self.linear3.weight)
        T.nn.init.ones_(self.linear1.bias)
        T.nn.init.ones_(self.linear2.bias)
        T.nn.init.ones_(self.linear3.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        
        x = self.linear2(x)
        x = F.leaky_relu(x)
        
        x = self.linear3(x)
        
        return x

class Encoder(T.nn.Module):
    '''Module that implements the encoder of the dual-head autoencoder
    
    Attributes
    ----------
    lstm1 : torch.nn.modules.rnn.LSTM
        Input layer
    lstm2 : torch.nn.modules.rnn.LSTM
        Hidden layer
    lstm3 : torch.nn.modules.rnn.LSTM
        Output layer, outputs the bottleneck
    dropout : torch.nn.modules.dropout.Dropout
        Dropout layer
    '''
    
    #'timeseries_n' (int) is the number of variables on the input of lstm1
    def __init__(self, timeseries_n):
        super().__init__()
        self.lstm1 = T.nn.LSTM(input_size = timeseries_n, 
                                hidden_size = math.ceil(timeseries_n*0.75),
                                num_layers=1,
                                dropout=0,
                                batch_first=True)
        self.lstm2 = T.nn.LSTM(input_size = math.ceil(timeseries_n*0.75), 
                                hidden_size = math.ceil(timeseries_n*0.5),
                                num_layers=1,
                                dropout=0,
                                batch_first=True)
        self.lstm3 = T.nn.LSTM(input_size = math.ceil(timeseries_n*0.5), 
                                hidden_size = math.ceil(timeseries_n*0.25),
                                num_layers=1,
                                dropout=0,
                                batch_first=True)                        
        self.dropout = T.nn.Dropout(p=0.3)
        
        #weight and bias initialization
        for lstm in [self.lstm1, self.lstm2, self.lstm3]:
            T.nn.init.kaiming_uniform_(lstm.weight_ih_l0)
            T.nn.init.kaiming_uniform_(lstm.weight_hh_l0)
            T.nn.init.ones_(lstm.bias_ih_l0)
            T.nn.init.ones_(lstm.bias_hh_l0)

    def forward(self, x):
        
        #batch size x sequence length x number of features 
        x = self.dropout(x)
        
        x, (hn, cn) = self.lstm1(x)
        rescon1 = x #residual connection 1 to decoder
        x, (hn, cn) = self.lstm2(x)
        rescon2 = x #residual connection 2 to decoder
        x, (hn, cn) = self.lstm3(x)
                
        return [x, rescon1, rescon2]
        
class Decoder(T.nn.Module):
    '''Module that implements the decoder of the dual-head autoencoder
    
    Attributes
    ----------
    lstm1 : torch.nn.modules.rnn.LSTM
        Input layer, takes the bottleneck of the encoder
    lstm2 : torch.nn.modules.rnn.LSTM
        Hidden layer
    lstm3 : torch.nn.modules.rnn.LSTM
        Output layer, outputs the bottleneck
    '''
    
    #'timeseries_n' (int) is the number of variables on the input of lstm1
    def __init__(self, timeseries_n):
        super().__init__()
        self.lstm1 = T.nn.LSTM(input_size = math.ceil(timeseries_n*0.25), 
                                hidden_size = math.ceil(timeseries_n*0.5),
                                num_layers=1,
                                dropout=0,
                                batch_first=True)
        self.lstm2 = T.nn.LSTM(input_size = math.ceil(timeseries_n*0.5), 
                                hidden_size = math.ceil(timeseries_n*0.75),
                                num_layers=1,
                                dropout=0,
                                batch_first=True)
        self.lstm3 = T.nn.LSTM(input_size = math.ceil(timeseries_n*0.75), 
                                hidden_size = timeseries_n,
                                num_layers=1,
                                dropout=0,
                                batch_first=True)
                                
        #weight and bias initialization
        for lstm in [self.lstm1, self.lstm2, self.lstm3]:
            T.nn.init.kaiming_uniform_(lstm.weight_ih_l0)
            T.nn.init.kaiming_uniform_(lstm.weight_hh_l0)
            T.nn.init.ones_(lstm.bias_ih_l0)
            T.nn.init.ones_(lstm.bias_hh_l0)
        
    def forward(self, x, rescon1, rescon2):
                
        x, (hn, cn) = self.lstm1(x)
        x = x + rescon2 #the residual connection 2 of the encoder arrives here
        x = F.leaky_relu(x)
        x, (hn, cn) = self.lstm2(x)
        x = x + rescon1 #the residual connection 1 of the encoder arrives here
        x = F.leaky_relu(x)
        x, (hn, cn) = self.lstm3(x)
        
        #print(x.size())
        
        return x

class DualHeadAutoencoder(T.nn.Module):
    '''Module that brings together the encoder, decoder and regressor head
    
    Attributes
    ----------
    encoder : Encoder
        encoder module
    decoder : Decoder
        decoder module
    regressor : NRRRegressor
        regressor module
    '''
    
    #'bottleneck_features' (int) is the number of features on the output of the encoder
    #'timesteps' (int) is the number of days in each timeseries
    #'timeseries_n' (int) is the number of variables on the input of lstm1
    #'scalars_n' (int) is the total number of scalars
    def __init__(self, bottleneck_features, timesteps, timeseries_n, scalars_n):
        super().__init__()
        self.encoder = Encoder(timeseries_n)
        self.decoder = Decoder(timeseries_n)
        self.regressor = NRRRegressor(bottleneck_features, timesteps, scalars_n)

    def forward(self, x_timeseries, x_scalars):
        [enc_out, rescon1, rescon2] = self.encoder(x_timeseries)
        dec_out = self.decoder(enc_out, rescon1, rescon2)
        
        reshaped_bottleneck = T.reshape(enc_out, (enc_out.shape[0], enc_out.shape[1]*enc_out.shape[2])) #flatten but keep batches
        regressor_input = T.cat((reshaped_bottleneck, x_scalars), 1) #concatenate the bottleneck with the scalars
        
        nrr_out = self.regressor(regressor_input)
        
        return [dec_out, nrr_out] #outputs the reconstructed input and the output of the regression head
    

def validate(device, net, dataloader, loss):
    '''Returns  list
    
    device: torch.device, cpu or gpu mode
    net: DualHeadAutoencoder, the network under validation
    dataloader: torch.utils.data.dataloader.DataLoader, the dataloader of the validation set
    loss: torch.nn.MSELoss, loss function

    Calculates the validation loss of each batch of an epoch and returns them 
    '''
    
    validation_losses = []
        
    with T.no_grad():    
            net.train(mode=False)
            for i, (x_timeseries, x_scalars, y_nrr, rest) in enumerate(dataloader):
                
                x_timeseries = x_timeseries.float().to(device)
                x_scalars = x_scalars.float().to(device)                
                y_nrr = y_nrr.float().to(device)               
                
                dec_out, nrr_out = net(x_timeseries, x_scalars)
                l = loss(dec_out, x_timeseries) + loss(nrr_out, y_nrr.unsqueeze(1))

                validation_losses.append(l.cpu())
                            
    return validation_losses

def train(device, net, train_dataloader, validation_dataloader, num_epochs, early_stopping_patience, tensorboard_writer):
    '''Returns a tuple containing the trained network, average training loss per epoch, average validation loss per epoch
    
    device: torch.device, cpu or gpu mode
    net: DualHeadAutoencoder, the network to be trained
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
    
    
    for epoch in range(num_epochs):
    
        net.train(mode=True)
        training_losses=[] 
        
        for i, (x_timeseries, x_scalars, y_nrr, rest) in enumerate(train_dataloader):
            
            x_timeseries = x_timeseries.float().to(device)
            x_scalars = x_scalars.float().to(device)
            y_nrr = y_nrr.float().to(device)       
            
            dec_out, nrr_out = net(x_timeseries, x_scalars)
            l = loss(dec_out, x_timeseries) + loss(nrr_out, y_nrr.unsqueeze(1))

            net.zero_grad()
            l.backward()
            optimizer.step()
            
            training_losses.append(l.detach().cpu())
            
        validation_losses = validate(net, validation_dataloader, loss)        
                        
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
            
def test(device, net, dataloader, type):
    '''Returns a tuple containing either the recostruction and regression rmses, or the reconstruction and regression rmses and a dataframe with the test data including the prediction for each sample 
    
    device: torch.device, cpu or gpu mode
    net: DualHeadAutoencoder, the network to be trained
    dataloader: torch.utils.data.dataloader.DataLoader, a training or test dataloader
    type: str, 'test' or 'training', if 'test' then it creates the test dataframe

    Evaluates the provided net
    '''
    
    df = pd.DataFrame(columns=['SoilFertility', 'Irrigation', 'FertMonth', 'FertDay', 'FertRate_orig', 'SoilWater_orig', 'Year', 'target_var', 'NRR', 'Residual'])
    results_tensor = T.tensor([])
        
    with T.no_grad():
        
        net.train(mode=False)
        
        errors_dec=np.array([])
        errors_nrr=np.array([])
        
        for i, (x_timeseries, x_scalars, y_nrr, rest) in enumerate(dataloader):

            #show progress but don't print all iterations
            if i>0 and i%500 == 0:
                print(f'Testing : iteration {str(i)}')
                                
            x_timeseries = x_timeseries.float().to(device)            
            x_scalars = x_scalars.float().to(device)
            y_nrr = y_nrr.float().to(device)       
                  
            #prediction = net(x)
            dec_pred, nrr_pred = net(x_timeseries, x_scalars)          
            
            squared_error_dec = (dec_pred-x_timeseries)**2
            squared_error_nrr = (nrr_pred-y_nrr)**2
 
            errors_dec = np.append(errors_dec, squared_error_dec.cpu())
            errors_nrr = np.append(errors_nrr, squared_error_nrr.cpu())
            
            if type=='test':
                results_tensor = T.cat((results_tensor, 
                                        T.cat((rest, 
                                                y_nrr.unsqueeze(1).cpu(),
                                                nrr_pred.cpu(),
                                                T.abs(y_nrr.unsqueeze(1).cpu()-nrr_pred.cpu())), dim=1)), dim = 0)
            
        error_dec = math.sqrt(errors_dec.mean())
        error_nrr = math.sqrt(errors_nrr.mean())
        
        if type=='test':
            return [error_dec, error_nrr, df.append(pd.DataFrame(results_tensor, columns=df.columns).astype('float'), ignore_index=True)]
        else:
            return [error_dec, error_nrr]

def run(device, scenario_id, batch_size, epochs, patience, bottleneck_features, timeseries_n, timesteps, scalars_n, train_clims, validation_clims, test_clim, metamodel_type, test_type, scenario_folder, filename_train, filename_validate, filename_test, irrigation, base_path, save_path):
    '''Returns None
    
    device: torch.device, cpu or gpu mode
    scenario_id: str, training scenario identifier
    batch_size: int, batch size for training validation test
    epochs: int, how many epochs to train
    patience: int, how many epochs to continue with consecutive validation losses above the lowest validation loss
    bottleneck_features: 
    timeseries_n
    timesteps: int, steps of the timeseries features
    scalars_n
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
    train_dataset = CustomDataset(training_df, timesteps)
    validation_dataset = CustomDataset(validation_df, timesteps)
    test_dataset =  CustomDataset(test_df, timesteps)
    print('Datasets read')
    
    #create Dataloader objects
    train_dataloader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers = True)
    validation_dataloader = DataLoader(dataset = validation_dataset, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers = True)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers = True) 
    
    #initialize tensorboard writer
    tensorboard_writer = SummaryWriter(comment = scenario_id)
    
    
    #initialize the net
    net = DualHeadAutoencoder(bottleneck_features, timesteps, timeseries_n, scalars_n)
    net.to(device)
    
    #train the net
    net, avg_train_losses, avg_valid_losses = train(device, net, train_dataloader, validation_dataloader, epochs, patience, tensorboard_writer)
    T.save(net.encoder.state_dict(), 'saved_encoder.pt')
    
    #load best network state
    checkpointed_net = DualHeadAutoencoder(bottleneck_features, timesteps, timeseries_n, scalars_n)
    checkpointed_net.load_state_dict(T.load('model_checkpoint.pt'))
    
    #show training error
    training_error_dec, training_error_nrr = test(device, checkpointed_net, train_dataloader, 'training')
    s_f_tr = scenario_id + ', ' + 'Checkpointed net train rmse:' + str(training_error_dec) + ' ' + str(training_error_nrr)
    print(s_f_tr)
    
    #show test error
    testing_error_dec, testing_error_nrr, testing_predictions_df = test(device, checkpointed_net, test_dataloader, 'test')
    s_f_te = scenario_id + ', ' + 'Checkpointed net test rmse:' + str(testing_error_dec) + ' ' + str(testing_error_nrr)
    print(s_f_te)

    #save predictions of test set
    if True:
        #add constant columns which will allow to compare to random forest
        testing_predictions_df['Metamodel type'] = metamodel_type
        testing_predictions_df['Test type'] = test_type
        testing_predictions_df['Location'] = test_clim
        
        create_path(save_path)
        testing_predictions_df.to_csv(save_path + '/predictions_testing_' + scenario_id + '_' + test_clim + '.csv', index=False)
    
    #plot 
    plot_training_validation_losses('autoencoder_' + scenario_id, epochs, avg_train_losses, avg_valid_losses)
    zip_files(scenario_id, glob.glob('*.txt') + glob.glob('*.png') + glob.glob('*.py') + glob.glob('*.pt'))


if __name__ == '__main__':
    device =  T.device('cuda' if T.cuda.is_available() else 'cpu')
    
    print(f'Started at {get_datetime()}')       
    
    base_path = '...'
    
    for train_clims, validation_clims, test_clim, metamodel_type, test_type, \
        scenario_folder, \
        filename_train, \
        filename_validate, \
        filename_test, \
        irrigation in [[['Clim5'], ['Clim5'], 'Clim5', 'local', 'known', \
                        'one_hot', \
                        'train_standardized_fert_soilwater.csv', \
                        'validation_standardized_fert_soilwater.csv', \
                        'test_standardized_fert_soilwater.csv', \
                        'both']]:
            
        print(f'Testing in: {test_clim}')
        print(f'Irrigation: {irrigation}')

        run(device = device,
            scenario_id = '...',
            batch_size = 32 , 
            epochs = 50,
            patience = 300,
            bottleneck_features = 4,
            timeseries_n = 15,
            timesteps = 28,
            scalars_n = 5,
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
            save_path = base_path + 'predictions/' + test_clim + '/' + scenario_folder + '/res_autoencoder_multitask_adamw_fert_soilwater_LSTM/' + test_type + '/' + metamodel_type + '/')

    print(f'Ended at {get_datetime()}')  




