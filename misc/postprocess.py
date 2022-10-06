import pandas as pd
from sklearn.preprocessing import StandardScaler
import os, glob, pickle, math, sys

def combine_dataset(path_to_folder):
    '''Returns  a pandas.core.frame.DataFrame
    
    path_to_folder: str, path where the output csvs from Spark processing are
    
    Combines the csvs of Spark processing into a single pandas dataframe
    '''     
    
    if os.path.isdir(path_to_folder):
        df = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(path_to_folder, "*.csv")))
    else:
        df = pd.read_csv(path_to_folder)
        
    return df

def standardize_values(train_df, validation_df, test_df, scaler):
    '''Returns  a list of pandas.core.frame.DataFrame and sklearn.preprocessing._data.StandardScaler
    
    train_df: pandas.core.frame.DataFrame, the training data dataframe
    validation_df: pandas.core.frame.DataFrame, the validation data dataframe
    test_df: pandas.core.frame.DataFrame, the test data dataframe
    scaler: sklearn.preprocessing._data.StandardScaler, the scaler provided. None means not provided
    
    Standardizes the training data, then the validation test data based on the training scaler. If a scaler is provided it standardizes everything based on it
    '''
    
    #seperate the case where we use observations with an already made scaler
    if scaler == None:
        scaler = StandardScaler()
        scaler.fit(train_df)
    else:
        scaler = pickle.load(open(scaler, 'rb'))
        
    train_df[train_df.columns] = scaler.transform(train_df)
    validation_df[validation_df.columns] = scaler.transform(validation_df)
    test_df[test_df.columns] = scaler.transform(test_df)    
        
    return [train_df, validation_df, test_df, scaler]

def add_month_to_x_y_columns(df):
    '''Returns  a pandas.core.frame.DataFrame
    
    df: pandas.core.frame.DataFrame, the dataframe were we want to encode the month
    
    Encodes the month variable into x, y coordinates
    '''
    
    #month to coordinates
    df['month_x'] = df['FertMonth'].apply(lambda month: math.cos(math.radians(month * 360/12)))
    df['month_y'] = df['FertMonth'].apply(lambda month: math.sin(math.radians(month * 360/12)))
    
    #round to zero, otherwise values that are almost zero appear but they are not treated as zero
    df['month_x'] = df['month_x'].apply(lambda x: 0 if abs(x) < 0.000001 else x)
    df['month_y'] = df['month_y'].apply(lambda y: 0 if abs(y) < 0.000001 else y)
    
    return df

def postprocess(clim, base_path, provided_scaler, scenario, train_csv, validation_csv, test_csv):
    '''Returns  None
    
    clim: str, the climate which is going to be processed
    base_path: str, root of the data and result folders
    provided_scaler: sklearn.preprocessing._data.StandardScaler, fitted scaler for standardization or None
    scenario: str, experiment identifier to know in which folder to save results
    train_csv: str, path to training data (csv or folder with csvs)
    validation_csv: str, path to training data (csv or folder with csvs)
    test_csv: str, (csv or folder with csvs)
    
    Adds 'FertRate_orig', 'SoilWater_orig' columns to preserve theses values after standardization, standardizes the data, converts months to x, y coordinates, and saves the resulting dataframes
    '''
    
    print('Doing climate', clim)

    train_path =      base_path + 'input/' + clim + '/' + train_csv
    validation_path = base_path + 'input/' + clim + '/' + validation_csv
    test_path =       base_path + 'input/' + clim + '/' + test_csv

    train_df = combine_dataset(train_path)
    train_df['FertRate_orig'] = train_df['FertRate']
    train_df['SoilWater_orig'] = train_df['SoilWater']
    validation_df = combine_dataset(validation_path)
    validation_df['FertRate_orig'] = validation_df['FertRate']
    validation_df['SoilWater_orig'] = validation_df['SoilWater']
    test_df = combine_dataset(test_path)
    test_df['FertRate_orig'] = test_df['FertRate']
    test_df['SoilWater_orig'] = test_df['SoilWater']

    initial_column_order = train_df.columns #we could take it also from validation or test set

    print('combined shapes')
    print(train_df.shape)
    print(validation_df.shape)
    print(test_df.shape)      
    
    #columns which will not be standardized
    cols_no_standardize = ['File', 'Weather', 'target_var',  
                           'SoilFertility',  'Irrigation', 'FertMonth', 'FertDay', 'FertRate_orig', 'SoilWater_orig', 'Year']

    train_df_file_target_var = train_df[cols_no_standardize]
    validation_df_file_target_var = validation_df[cols_no_standardize]
    test_df_file_target_var = test_df[cols_no_standardize]             
                 
    standardized_train_df, standardized_validation_df, standardized_test_df, scaler = standardize_values(train_df.drop(columns=cols_no_standardize), validation_df.drop(columns=cols_no_standardize), test_df.drop(columns=cols_no_standardize), provided_scaler)

    #save scaler
    #we only want to save it if it's created this time. If it's already provided then there is no point in saving the same thing again
    if provided_scaler is None:
        pickle.dump(scaler, open(base_path + 'input/' + clim + '/scaler' + '_' + clim + '_' + scenario + '_' + train_csv[:-4], 'wb'))

    print('standardized shapes')
    print(standardized_train_df.shape)
    print(standardized_validation_df.shape)
    print(standardized_test_df.shape) 

    # put non-standardized columns back to the df that has the standardized ones
    for col in cols_no_standardize:
        standardized_train_df[col] = train_df_file_target_var[col]
        standardized_validation_df[col] = validation_df_file_target_var[col]
        standardized_test_df[col] = test_df_file_target_var[col]

    # the commented lines can replace the loop above but they show: numpy.core._exceptions.MemoryError: Unable to allocate 8.34 GiB for an array with shape (423, 2646130) and data type float64
    # standardized_train_df = standardized_train_df.join(train_df_file_target_var)
    # standardized_validation_df = standardized_validation_df.join(validation_df_file_target_var)
    # standardized_test_df = standardized_test_df.join(test_df_file_target_var)

    #rearrange columns
    standardized_train_df =      standardized_train_df[      initial_column_order]
    standardized_validation_df = standardized_validation_df[ initial_column_order]
    standardized_test_df =       standardized_test_df[       initial_column_order]

    print('standardized shapes with File and target_var')
    print(standardized_train_df.shape)
    print(standardized_validation_df.shape)
    print(standardized_test_df.shape)

    #add x y representation for month
    standardized_train_df = add_month_to_x_y_columns(standardized_train_df)
    standardized_validation_df = add_month_to_x_y_columns(standardized_validation_df)
    standardized_test_df = add_month_to_x_y_columns(standardized_test_df)
    
    #rearrange columns again
    columns = list(standardized_train_df.columns)
    columns = columns[:-8] + columns[-2:] + [columns[-8]] + columns[-6:-2]
    standardized_train_df = standardized_train_df[columns]
    standardized_validation_df = standardized_validation_df[columns]
    standardized_test_df = standardized_test_df[columns]

    #save dfs
    standardized_train_df.to_csv(      base_path + 'input/' + clim + '/' + train_csv[:-4] + '_standardized_fert_soilwater' + '_' + scenario + '.csv', index=False)
    standardized_validation_df.to_csv( base_path + 'input/' + clim + '/' + validation_csv[:-4] + '_standardized_fert_soilwater' + '_' + scenario + '.csv', index=False)
    standardized_test_df.to_csv(       base_path + 'input/' + clim + '/' + test_csv[:-4] + '_standardized_fert_soilwater' + '_' + scenario + '.csv', index=False)
            


def run():
    clim = 'Clim7'
    train_csv = 'train.csv'
    validation_csv = 'validation.csv'
    test_csv = 'test.csv'
    base_path = '...'
    scaler = None
    scenario = 'only_SoilFertility1'

    postprocess(clim, base_path, scaler, scenario, train_csv, validation_csv, test_csv)    

#with the below we can write 'python postprocess_standardize_fert_soilwater_for_lstm.py run'
if __name__ == '__main__':
    globals()[sys.argv[1]]()

