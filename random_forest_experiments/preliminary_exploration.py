from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pandas as pd
import os
import glob
import math

def read_data(training_path, test_path):
    '''Returns a list of pandas.core.frame.DataFrame
    
    training_path: st, training data path
    test_path: st, test data path

    Takes the training and test data paths and returns the corresponding dataframes.
    '''
    
    trainingDF = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(training_path, '*.csv')))
    testDF = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(test_path, '*.csv')))
    
    return [trainingDF, testDF]

def standardize_values(train_df, test_df):
    '''Returns a list of pandas.core.frame.DataFrame
    
    training_path: pyspark.sql.dataframe.DataFrame, dataframe containing the training data
    test_path: pyspark.sql.dataframe.DataFrame, dataframe containing the test data

    Standardizes the training data, and then the test data with the same scaler.
    '''
    
    scaler = StandardScaler()
    scaler.fit(train_df)
    train_df[train_df.columns] = scaler.transform(train_df)
    test_df[test_df.columns] = scaler.transform(test_df)
    
    return [train_df, test_df]

def run():   
    '''Returns None

    Tests the performance of several algorithms on some fixed parameters.
    '''
    
    print('reading data')
    trainingDF, testDF = read_data('...', '...')
    
    print('standardize data')
    train_X, test_X = standardize_values(trainingDF.drop(['Weather', 'Year', 'FertDay', 'target_var'], axis=1), 
                                            testDF.drop(['Weather', 'Year', 'FertDay', 'target_var'], axis=1))
    
    train_Y = trainingDF['target_var']
    test_Y = testDF['target_var']
    
    models = [ElasticNet, RandomForestRegressor, MLPRegressor, GradientBoostingRegressor, LinearSVR, SVR]

    parameters = [{'alpha':[0.2, 0.5, 1], 'max_iter':[500, 1000, 2000], 'l1_ratio':[0.2, 0.5, 0.8]},                
                 {'n_estimators':[100, 200], 'max_depth':[3, 7, 12], 'min_samples_split':[10, 20], 'min_samples_leaf':[10, 30], 'max_features':[0.33], 'n_jobs':[25]},
                 {'hidden_layer_sizes':[(50,), (60,60)], 'activation':['relu'], 'batch_size':[32], 'max_iter':[100], 'early_stopping':[True], 'n_iter_no_change':[20]},
                 {'learning_rate':[0.05, 0.1, 0.2], 'n_estimators':[100, 200], 'min_samples_split':[10, 20], 'min_samples_leaf':[10, 30],'max_depth':[3, 7, 12], 'max_features':[0.33]},
                 {'C':[0.2, 0.5, 1], 'epsilon':[0.05, 0.1, 0.2], 'loss':['epsilon_insensitive', 'squared_epsilon_insensitive']},
                 {'kernel':['rbf'], 'C':[0.2, 0.5, 1], 'epsilon':[0.05, 0.1, 0.2], 'cache_size':[2000]}]

    grids = []

    for model, params, name in zip(models, parameters, ['Elastic_net', 'RF', 'MLP', 'GB', 'LinearSVR', 'SVR']):
        print('gridsearch', name)
        grid = GridSearchCV(estimator=model(), param_grid=params, scoring='neg_root_mean_squared_error', n_jobs=15)
        grid.fit(train_X, train_Y)
        grids.append(grid)
        print(name, grid.best_params_)
        y_pred = grid.predict(test_X)
        print(name, math.sqrt(mean_squared_error(test_Y, y_pred)))
        dump(grid, name + '.joblib')

    print('All done')    