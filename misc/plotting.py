import matplotlib, os, numpy as np    
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from helper_functions import get_datetime

def plot_training_validation_losses(scenario_id, num_epochs, avg_train_losses, avg_valid_losses):
    '''Returns None 
    
    scenario_id: str, scenario identifier
    num_epochs: int, how many epochs training lasted
    avg_train_losses: list, average training losses per epoch
    avg_valid_losses: list, average validation losses per epoch
    
    Creates a plot that shows the training and validation losses, and marks the point with the lowest validaiton loss.
    '''     
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.plot(range(1,len(avg_train_losses)+1),avg_train_losses, label='Training Loss')
    ax.plot(range(1,len(avg_valid_losses)+1),avg_valid_losses, label='Validation Loss')
    
    # find epoch of lowest validation loss
    minposs_validation = avg_valid_losses.index(min(avg_valid_losses))+1 
    ax.axvline(minposs_validation, linestyle='--', color='r',label='Min validation loss')
    ax.axhline(min(avg_valid_losses), linestyle='--', color='r')
    
    # find epoch of lowest training loss
    minposs_train = avg_train_losses.index(min(avg_train_losses))+1 
    ax.axvline(minposs_train, linestyle='--', color='g',label='Min training loss')
    ax.axhline(min(avg_train_losses), linestyle='--', color='g')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(scenario_id + '_' + get_datetime() + '.png')
    plt.close('all')
    
def make_boxplots(known_unknown, totalDF, x_axis, irrigation, test_location_names, scenario_folder):
    '''Returns None 
    
    known_unknown: str, one of 'known' 'unknown'
    totalDF: pandas.core.frame.DataFrame, contains the predictions for all models, model types, and location types and also the correspodning simulation parameters
    x_axis: str, one of 'FertMonth' 'Year'
    irrigation: int, one of '0' '1' 
    
    Creates plots of monthly and yearly residuals and saves them.
    '''     
    
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(18, 8) if x_axis=='FertMonth' else (14, 8))
    #if x_axis == 'FertMonth':
    palette = 'Set3'
    #else:
    #    palette = 'Set2'

    for index, (ax, location, irrigation) in enumerate(zip(axes.flatten(), ['Waiotu', 'Mahana'], [None, None])):    
        
        if irrigation is None:
            p=sns.boxplot(ax=ax, data=totalDF[(totalDF['Test type']==known_unknown) & (totalDF['Test location']==location)], x=x_axis, y='Residual', hue='Algorithm', palette=palette, showfliers = False)
        else:
            print(known_unknown, x_axis, irrigation, location, scenario_folder)
            p=sns.boxplot(ax=ax, data=totalDF[(totalDF['Test type']==known_unknown) & (totalDF['Test location']==location) & (totalDF['Irrigation']==irrigation)], x=x_axis, y='Residual', hue='Algorithm', palette=palette, showfliers = False)
        
        for label in p.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        ax.legend_.remove()
        p.set_title(location, fontdict= {'fontsize': 24})
        p.set(ylim=(-1, 14))
        p.set(yticks=np.arange(0, 15, 2.5))
        p.set(xlabel='')
        p.set(ylabel='')
        ax.tick_params(labelsize=26)
        ax.axhline(5, linestyle='--', color='dimgray')
    
    fig.text(0.48, 0.01, 'Month' if x_axis == 'FertMonth' else 'Year', ha='center', fontsize=28)
    if x_axis=='Year':
       fig.text(0.025, 0.5, 'Residual', va='center', rotation='vertical', fontsize=28)
    else:
       fig.text(0.05, 0.5, 'Residual', va='center', rotation='vertical', fontsize=28)
      
    # Create the legend
    axLine, axLabel = axes[0].get_legend_handles_labels()
    
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(axLine, axLabel, loc = 'center right', fontsize=18)

    # Adjust the scaling factor to fit your legend text completely outside the plot
    # (smaller value results in more space being made for the legend)
    plt.subplots_adjust(right=0.775,
                        wspace=0.05, 
                        hspace=0.12)
    
    plt.savefig(known_unknown.capitalize() + '_' + x_axis + '_' + get_datetime() + '.png', transparent=True)
    plt.close()
 
  
def combine_known_unknown_predictions(prediction_path, scenario_folder, algorithms, test_locations, known_unknown, model_types):
    '''Returns a pandas.core.frame.DataFrame
    
    prediction_path: str, the path where the results are saved
    scenario_folder: str, this along with 'predictions_base_path' will point to the folder where the predictions are
    
    Combines the predictions for all models, model types,  location types and simulation parameters
    '''
    totalDF = pd.DataFrame()
    for algorithm in algorithms:
        for location_type in known_unknown:
            for test_location in  test_locations:
                for model_type in model_types:     
                
                    predictionsDF = pd.concat(pd.read_csv(f) for f in [
                                                                        prediction_path + test_location + scenario_folder + algorithm + '/' + location_type + '/' + model_type + '/predictions_testing_seed_1_' + test_location + '.csv',
                                                                        prediction_path + test_location + scenario_folder + algorithm + '/' + location_type + '/' + model_type + '/predictions_testing_seed_32_' + test_location + '.csv',
                                                                        prediction_path + test_location + scenario_folder + algorithm + '/' + location_type + '/' + model_type + '/predictions_testing_seed_86_' + test_location + '.csv',
                                                                        prediction_path + test_location + scenario_folder + algorithm + '/' + location_type + '/' + model_type + '/predictions_testing_seed_1001_' + test_location + '.csv',
                                                                        prediction_path + test_location + scenario_folder + algorithm + '/' + location_type + '/' + model_type + '/predictions_testing_seed_40000_' + test_location + '.csv'])
                    
                    print('Algorithm:', algorithm, ' , irrigation:', predictionsDF.Irrigation.unique())
                    #print('Soil fertility', predictionsDF.SoilFertility.unique())
                    #print('Soil water', predictionsDF.SoilWater.unique())
                    #print('Fertrate', predictionsDF.FertRate.unique())
                    #print('---------------------------------------------')
                
                
                    df = pd.DataFrame()
                    df['Year'] = predictionsDF['Year'].astype(int)
                    df['FertMonth'] = predictionsDF['FertMonth'].astype(int)
                    df['target_var'] = predictionsDF['target_var']
                    if 'NRR' in predictionsDF:
                        df['NRR'] = predictionsDF['NRR'] #which means that was created by a NN df
                        df['Metamodel type'] = predictionsDF['Metamodel type']
                        df['Test type'] = predictionsDF['Test type']
                        df['Residual'] = predictionsDF['Residual']
                    else:
                        df['NRR'] = predictionsDF[model_type + '_model'] #which means that this was created by a rf df
                        df['Metamodel type'] = model_type
                        df['Test type'] = location_type
                        df['Residual'] = (df['target_var'] - df['NRR']).abs()                               
                    df['Test location'] = test_location
                    df['Irrigation'] = predictionsDF['Irrigation'].astype(int)
                    df['Algorithm'] = {'rf_seed':'Random Forest',
                                        'simple_MLP_adamw_fert_soilwater':'Multilayer Perceptron',
                                        'res_autoencoder_transfer_adamw_fert_soilwater':'Autoencoder',
                                        'res_autoencoder_multitask_adamw_fert_soilwater':'Dual-head autoencoder'}[algorithm]
                    
                    totalDF = totalDF.append(df, ignore_index=True)
        
    totalDF['Test location'] = totalDF['Test location'].map({'Clim1':'Waiotu', 'Clim2':'Ruakura', 'Clim3':'Wairoa', 'Clim4':'Marton', 'Clim5':'Mahana', 'Clim6':'Kokatahi', 'Clim7':'Lincoln', 'Clim8':'Wyndham'})
        
    print(totalDF)    
        
    return totalDF         

def run():   
    '''Returns None
    
    Makes boxplots for the provided algorithms and locations.
    '''
    
    prediction_path = '...'
    scenario_folder = 'one_hot'
    algorithms = ['rf_seed', 
                'simple_MLP_adamw_fert_soilwater',
                'res_autoencoder_transfer_adamw_fert_soilwater',
                'res_autoencoder_multitask_adamw_fert_soilwater']
                
    known_unknown = ['known']
    test_locations = ['Clim1', 'Clim5']
    test_location_names = list(map(lambda test_location: {'Clim1':'Waiotu', 'Clim2':'Ruakura', 'Clim3':'Wairoa', 'Clim4':'Marton',
                                                        'Clim5':'Mahana', 'Clim6':'Kokatahi', 'Clim7':'Lincoln', 'Clim8':'Wyndham'}[test_location], test_locations))
    only_non_irrigated = False
    model_types = ['local']
    
    #Combine dataframes
    combined_predictionsDF = combine_known_unknown_predictions(prediction_path, scenario_folder, algorithms, test_locations, known_unknown, model_types)
    
    #Make plots
    for location_type in known_unknown:
        for x_axis in ['FertMonth', 'Year']:
            #if only_non_irrigated==False:
                #make_boxplots(location_type, combined_predictionsDF, x_axis, None, test_location_names, scenario_folder)
            
            make_boxplots(location_type, combined_predictionsDF, x_axis, '-', test_location_names, scenario_folder)
            
    print('Residual plotting done')  