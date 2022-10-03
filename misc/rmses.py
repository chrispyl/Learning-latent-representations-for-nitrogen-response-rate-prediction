import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from math import sqrt
from functools import reduce

base_path = '...'

for clim in ['Clim1', 'Clim5']:
    
    print(clim)

    rf_df = pd.concat(pd.read_csv(f) for f in [
    base_path + 'predictions/' + clim + '/one_hot/rf_seed/known/local/predictions_testing_seed_1_' + clim + '.csv',
    base_path + 'predictions/' + clim + '/one_hot/rf_seed/known/local/predictions_testing_seed_32_' + clim + '.csv',
    base_path + 'predictions/' + clim + '/one_hot/rf_seed/known/local/predictions_testing_seed_86_' + clim + '.csv',
    base_path + 'predictions/' + clim + '/one_hot/rf_seed/known/local/predictions_testing_seed_1001_' + clim + '.csv',
    base_path + 'predictions/' + clim + '/one_hot/rf_seed/known/local/predictions_testing_seed_40000_' + clim + '.csv'])

    mlp_df = pd.concat(pd.read_csv(f) for f in [
    base_path + 'predictions/' + clim + '/one_hot/simple_MLP_adamw_fert_soilwater/known/local/predictions_testing_seed_1_' + clim + '.csv',
    base_path + 'predictions/' + clim + '/one_hot/simple_MLP_adamw_fert_soilwater/known/local/predictions_testing_seed_32_' + clim + '.csv',
    base_path + 'predictions/' + clim + '/one_hot/simple_MLP_adamw_fert_soilwater/known/local/predictions_testing_seed_86_' + clim + '.csv',
    base_path + 'predictions/' + clim + '/one_hot/simple_MLP_adamw_fert_soilwater/known/local/predictions_testing_seed_1001_' + clim + '.csv',
    base_path + 'predictions/' + clim + '/one_hot/simple_MLP_adamw_fert_soilwater/known/local/predictions_testing_seed_40000_' + clim + '.csv'])
    
    autoencoder_df = pd.concat(pd.read_csv(f) for f in [
    base_path + 'predictions/' + clim + '/one_hot/res_autoencoder_transfer_adamw_fert_soilwater/known/local/predictions_testing_seed_1_' + clim + '.csv',
    base_path + 'predictions/' + clim + '/one_hot/res_autoencoder_transfer_adamw_fert_soilwater/known/local/predictions_testing_seed_32_' + clim + '.csv',
    base_path + 'predictions/' + clim + '/one_hot/res_autoencoder_transfer_adamw_fert_soilwater/known/local/predictions_testing_seed_86_' + clim + '.csv',
    base_path + 'predictions/' + clim + '/one_hot/res_autoencoder_transfer_adamw_fert_soilwater/known/local/predictions_testing_seed_1001_' + clim + '.csv',
    base_path + 'predictions/' + clim + '/one_hot/res_autoencoder_transfer_adamw_fert_soilwater/known/local/predictions_testing_seed_40000_' + clim + '.csv'])
    
    dual_autoencoder_df = pd.concat(pd.read_csv(f) for f in [
    base_path + 'predictions/' + clim + '/one_hot/res_autoencoder_multitask_adamw_fert_soilwater/known/local/predictions_testing_seed_1_' + clim + '.csv',
    base_path + 'predictions/' + clim + '/one_hot/res_autoencoder_multitask_adamw_fert_soilwater/known/local/predictions_testing_seed_32_' + clim + '.csv',
    base_path + 'predictions/' + clim + '/one_hot/res_autoencoder_multitask_adamw_fert_soilwater/known/local/predictions_testing_seed_86_' + clim + '.csv',
    base_path + 'predictions/' + clim + '/one_hot/res_autoencoder_multitask_adamw_fert_soilwater/known/local/predictions_testing_seed_1001_' + clim + '.csv',
    base_path + 'predictions/' + clim + '/one_hot/res_autoencoder_multitask_adamw_fert_soilwater/known/local/predictions_testing_seed_40000_' + clim + '.csv'])
    
    rf_df = rf_df.rename(columns={'local_model': 'rf_nrr'})
    mlp_df = mlp_df.rename(columns={'NRR': 'mlp_nrr', 'SoilWater_orig': 'SoilWater', 'FertRate_orig': 'FertRate'})
    autoencoder_df = autoencoder_df.rename(columns={'NRR': 'auto_nrr', 'SoilWater_orig': 'SoilWater', 'FertRate_orig': 'FertRate'})
    dual_autoencoder_df = dual_autoencoder_df.rename(columns={'NRR': 'auto_dual_nrr', 'SoilWater_orig': 'SoilWater', 'FertRate_orig': 'FertRate'})

    #print(rf_df.dtypes)
    #print(mlp_df.dtypes)
    
    target_var = lambda x: round(x, 3)
    
    #mlp_df['SoilWater'] = mlp_df['SoilWater'].apply(soilwater)
    #mlp_df['FertRate'] = mlp_df['FertRate'].apply(fertrate)
    rf_df['target_var'] = rf_df['target_var'].apply(target_var)
    mlp_df['target_var'] = mlp_df['target_var'].apply(target_var)
    autoencoder_df['target_var'] = autoencoder_df['target_var'].apply(target_var)
    dual_autoencoder_df['target_var'] = dual_autoencoder_df['target_var'].apply(target_var)
    
    
    for df in [mlp_df, autoencoder_df, dual_autoencoder_df]:
        df['Irrigation'] = df['Irrigation'].astype('int64')
        df['SoilWater'] = df['SoilWater'].astype('int64')
        df['FertDay'] = df['FertDay'].astype('int64')
        df['FertMonth'] = df['FertMonth'].astype('int64')
        df['Year'] = df['Year'].astype('int64')
        df['SoilFertility'] = df['SoilFertility'].astype('int64')
        df['FertRate'] = df['FertRate'].astype('int64')

    #print(mlp_df)
    #print(rf_df)
    
    #print(rf_df.dtypes)
    #print(mlp_df.dtypes)

    merged_df = reduce(lambda left,right: pd.merge(left, right, on=['target_var', 'SoilWater', 'SoilFertility', 'Irrigation', 'FertMonth', 'FertRate', 'FertDay', 'Year']), [rf_df, mlp_df, autoencoder_df, dual_autoencoder_df])

    #print(merged_df.columns)
    print('Clim', clim)
    print(merged_df.std(ddof=0))
    
    for error in ['rf_nrr', 'mlp_nrr', 'auto_nrr', 'auto_dual_nrr']:
        print(merged_df[error].mad())
    
    for dataframe in [merged_df]:
            for metric, name in [[mean_absolute_error, 'mae'], [r2_score, 'r2']]:
                print(name)
                for nrr in ['rf_nrr', 'mlp_nrr', 'auto_nrr', 'auto_dual_nrr']:
                    print(round(metric(dataframe['target_var'], dataframe[nrr]), 2))
            