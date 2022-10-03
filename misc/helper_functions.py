import os
import datetime    
import zipfile
import pandas as pd

def get_dfs_from_clims(base_path, train_clims, validation_clims, test_clim, filename_train, filename_validate, filename_test, irrigation=None):
    '''Returns a tuple containing 3 pandas.core.frame.DataFrame
    
    base_path: str, a path where input data, predictions or other stuff are stored
    train_clims: list, contains the names of the training climates
    validation_clims: list, contains the names of the validation climates
    test_clim: str, name of the test climate
    filename_train: str, name of the file containing the training data for each climate (assumes that in all climates the filename is the same)
    filename_validate: str, name of the file containing the validation data for each climate (assumes that in all climates the filename is the same)
    filename_test: str, name of the file containing the test climate data
    irrigation: str, shows which data to keep, either 'irrigated', 'non_irrigated', or else keeps both
    
    Returns the closest 'n' locations to each of the eight available locations, ordered based on increasing distance
    '''
    
    training_paths = [base_path + 'input/' + clim + '/' + filename_train for clim in train_clims]
    validation_paths = [base_path + 'input/' + clim + '/' + filename_validate for clim in validation_clims]
    test_path = base_path + 'input/' + test_clim + '/' + filename_test
        
    if irrigation=='irrigated':
        training_df = pd.concat(pd.read_csv(f) for f in training_paths).loc[lambda df: df.Irrigation == 1].drop('Irrigation', axis=1)
        validation_df = pd.concat(pd.read_csv(f) for f in validation_paths).loc[lambda df: df.Irrigation == 1].drop('Irrigation', axis=1)
        test_df = pd.read_csv(test_path).loc[lambda df: df.Irrigation == 1].drop('Irrigation', axis=1)
    elif irrigation=='non_irrigated':
        training_df = pd.concat(pd.read_csv(f) for f in training_paths).loc[lambda df: df.Irrigation == 0].drop('Irrigation', axis=1)
        validation_df = pd.concat(pd.read_csv(f) for f in validation_paths).loc[lambda df: df.Irrigation == 0].drop('Irrigation', axis=1)
        test_df = pd.read_csv(test_path).loc[lambda df: df.Irrigation == 0].drop('Irrigation', axis=1)
    else:
        #keep irrigated and non-irigated
        training_df = pd.concat(pd.read_csv(f) for f in training_paths)
        validation_df = pd.concat(pd.read_csv(f) for f in validation_paths)
        test_df = pd.read_csv(test_path)
    
    return training_df, validation_df, test_df

def create_path(path):  
    '''Returns None
    
    path: str, path to be created
    
    Creates the supplied path if it doesn't exist
    '''
    try:
        os.makedirs(path)
    except:
        print(path + ' exists')    

def zip_files(scenario_id, files_to_zip):
    '''Returns None
    
    scenario_id: str, scenario identifier to be inserted to the name of the zip
    files_to_zip: list, filetypes to be zipped
    
    Zippes files based on the supplied filetypes into an archive and names into based into scenario_id
    '''
    with zipfile.ZipFile(scenario_id + '.zip', mode='w') as zf:
        for filename in files_to_zip:
            zf.write(filename)

def get_datetime():
    '''Returns str 
    
    Returns the current date, replacing ':' in order to be able to be used as a file name
    '''
    date = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    date = date.replace(':', '')
    
    return date

def log_to_tensorboard(tensorboard_writer, net, epoch, train_loss, valid_loss):
    '''Returns None
    
    tensorboard_writer: torch.utils.tensorboard.writer.SummaryWriter obejct 
    net: a subclass of nn.Module, the neural network to be monitored 
    epoch: int, current epoch 
    train_loss: double, training loss for this epoch
    valid_loss: double, validation loss for this epoch
    
    Monitors weights and biases of the supplied neural network
    '''
    #mlp1_hid 1 & 2
    # tensorboard_writer.add_histogram('mlp1_hid1.bias', net.mlp1_hid1.bias, epoch)
    # tensorboard_writer.add_histogram('mlp1_hid1.weight', net.mlp1_hid1.weight, epoch)
    # tensorboard_writer.add_histogram('mlp1_hid2.bias', net.mlp1_hid2.bias, epoch)
    # tensorboard_writer.add_histogram('mlp1_hid2.weight', net.mlp1_hid2.weight, epoch)
    
    #mlp2_hid 1 & 2
    # tensorboard_writer.add_histogram('mlp2_hid1.bias', net.mlp2_hid1.bias, epoch)
    # tensorboard_writer.add_histogram('mlp2_hid1.weight', net.mlp2_hid1.weight, epoch)
    # tensorboard_writer.add_histogram('mlp2_hid2.bias', net.mlp2_hid2.bias, epoch)
    # tensorboard_writer.add_histogram('mlp2_hid2.weight', net.mlp2_hid2.weight, epoch)
    
    #losses
    tensorboard_writer.add_scalars('Epoch loss', {'train_loss':train_loss, 'validation_loss':valid_loss}, epoch)


    