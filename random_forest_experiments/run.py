import preprocessing_clover
import ML_clover
import helper_functions
import plotting
import os
import shutil

def main():
    '''
    Returns None
    
    Orchestrates the data processing and machine learning procedures for experimenting with random forest
    '''
    #DEFINE PATHS
    simulations_path = '...'
    weather_path = '...'
    nitrogen_path = '...'
    
    scenario_name = '...'
    preprocessing_results_path = '...' + scenario_name
    
    ml_results_path = '...'
    helper_functions.create_path(ml_results_path + '/' + scenario_name + '/' + 'known')
    helper_functions.create_path(ml_results_path + '/' + scenario_name + '/' + 'unknown')
        
    #PREPROCESSING
    preprocessing_clover.preprocessing_pipeline(False,
                                                [7],
                                                -28,
                                                -1,
                                                ['AboveGroundWt', 'NetGrowthWt', 'NetPotentialGrowthWt', 'SoilWater300', 'SoilTemp300', 'SoilTemp050', 'AboveGroundNConc'],
                                                True,
                                                True,
                                                simulations_path,
                                                weather_path,
                                                nitrogen_path,
                                                preprocessing_results_path)
    
    print('Preprocessing done')

    #ML
    for seed in [1, 32, 86, 1001, 40000]:
        for location_type in ['known']:           
            for irrigation in ['both']:
                ML_clover.ml_pipeline(scenario_name + '_' + location_type,
                         location_type,
                         'target_var',
                         preprocessing_results_path,
                         irrigation,
                         seed)
        
        move .png .csv .xlsx .model files to another directory
        os.system('find . \( -name "*.png" -o -name "*.csv" -o -name "*.xlsx" -o -name "*.model" \) -exec mv "{}" ' + ml_results_path + '/' + scenario_name + '/' + location_type + ' \;')
    
    print('ML done')
    
    #RESIDUAL PLOTTING
    combined_predictionsDF = helper_functions.combine_known_unknown_predictions(ml_results_path, scenario_name)
    
    for location_type in ['known', 'unknown']:
        for x_axis in  ['FertMonth', 'Year']:
            plotting.make_boxplots(location_type, combined_predictionsDF, x_axis)
            for irrigation in [0, 1]:
                plotting.make_boxplots(location_type, combined_predictionsDF, x_axis, irrigation)
         
        #move .png .csv .xlsx .model files to another directory
        os.system('find . -name "*.png" -exec mv "{}" ' + ml_results_path + '/' + scenario_name + '/' + location_type + ' \;')
    
    print('Residual plotting done')

    #ARCHIVING  
    shutil.make_archive(ml_results_path + '/' + scenario_name + '_irrigation_0', 'zip', ml_results_path + '/' + scenario_name)
    
    #CLEANING UP
    os.system('rm -rf ' + ml_results_path + '/' + scenario_name)
    
    print('All tasks finished')
  

if __name__=='__main__':
    main()
