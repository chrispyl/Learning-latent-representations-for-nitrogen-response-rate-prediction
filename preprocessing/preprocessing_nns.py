from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F
import schemas
import re
from natsort import natsorted

def combine_input_datasets(WeatherDataAllDF, simulationsDF, biophysicals):
    '''Returns a pyspark.sql.dataframe.DataFrame
    
    WeatherDataAllDF: pyspark.sql.dataframe.DataFrame, dataframe containing the weather data
    simulationsDF: pyspark.sql.dataframe.DataFrame, dataframe containing the simulations
    biophysicals: list, biophysical variables of the simulations
    
    Creates a dataframe with the simulations output and the corresponding weather for each simulation. Only the biophysical variables contained in 'biophysicals' are preserved from the simulations dataframe.
    '''
    
    WeatherDataAllDF = WeatherDataAllDF.select(*[column for column in WeatherDataAllDF.columns if (column != 'Date' and column != 'Weather')],
                                               F.to_date(
                                                   F.col("Date"), "dd/MM/yyyy").alias('Date'),
                                               F.substring(F.col('Weather'), 0, 5).alias('Weather'))

    simulationsDF = simulationsDF.select(*['File', 'FertiliserApplied', 'HerbageCut'],
                                         *biophysicals,
                                         F.to_date(
                                             F.col("Date"), "dd/MMM/yyyy").alias('Date'),
                                         F.substring(F.col('File'), -5, 5).alias('Weather'),
                                         F.col('DaysRelative').cast("Int").alias('DaysRelative'))

    return simulationsDF.join(WeatherDataAllDF, ['Date', 'Weather'])

def extract_columns_from_filename(dataDF):
    '''Returns a pyspark.sql.dataframe.DataFrame
    
    dataDF: pyspark.sql.dataframe.DataFrame, a dataframe containing the weather and simulation data
    
    Extracts the simulation parameters from the 'File' column strings and adds them as columns.
    '''
    dataDF = dataDF.withColumn('parts', F.split(F.col('File'), '_'))\
                   .withColumn('SoilFertility', F.substring(F.element_at('parts', 6), 9, 4))\
                   .withColumn('Irrigation', F.substring(F.element_at('parts', 7), 11, 3))\
                   .select(*dataDF.columns,
                           F.substring(F.element_at('parts', 1), 2, 4).cast(
                               IntegerType()).alias('Year'),
                           F.substring(F.element_at('parts', 2), 2, 2).cast(
                               IntegerType()).alias('FertMonth'),
                           F.substring(F.element_at('parts', 3), 2, 2).cast(
                               IntegerType()).alias('FertDay'),
                           F.substring(F.element_at('parts', 4), 11, 3).cast(
                               IntegerType()).alias('FertRate'),
                           F.substring(F.element_at('parts', 5), 5, 3).cast(
                               IntegerType()).alias('SoilWater'),
                           F.when(F.col('SoilFertility') == 'Low', 1).otherwise(
                               F.when(F.col('SoilFertility') == 'Med', 2).otherwise(3)).alias('SoilFertility'),
                           F.when(F.col('Irrigation') == 'Off', 0).otherwise(F.when(F.col('Irrigation') == 'On', 1)).alias('Irrigation'))

    return dataDF


def calculate_baseline_herbage_cuts(dataDF):
    '''Returns a pyspark.sql.dataframe.DataFrame
    
    dataDF: pyspark.sql.dataframe.DataFrame, a dataframe containing the weather and simulation data
    
    Calculates the herbage buts for the baseline case and adds them to the supplied dataframe.
    '''
    
    baseline_herbagecut_aggDF = dataDF.filter(F.col('FertRate') == 0)\
                                      .groupBy(['Weather', 'Year', 'SoilWater',
                                               'SoilFertility', 'Irrigation', 'FertMonth',
                                                'FertDay'])\
                                      .agg(F.sum(
                                          F.when(
                                              # take the sum in 0<days<65 to include the 2 cuts
                                              (F.col('DaysRelative') >= 0) & (
                                                  F.col('DaysRelative') <= 65),
                                              F.col('HerbageCut'))).alias('baseline_0_65_HerbageCut'))

    dataDF = dataDF.join(baseline_herbagecut_aggDF, [
                         'Weather', 'Year', 'SoilWater', 'SoilFertility', 'Irrigation', 'FertMonth', 'FertDay'])

    return dataDF


def add_target_variable(dataDF, NitrogenResponseRateDF):
    '''Returns a pyspark.sql.dataframe.DataFrame
    
    dataDF: pyspark.sql.dataframe.DataFrame, a dataframe containing the weather and simulation data
    NitrogenResponseRateDF: pyspark.sql.dataframe.DataFrame, a dataframe containing the nitrogen response rates of the simulations
    
    Adds the nitrogen response rate column to the supplied dataframe
    '''
    
    # put target variable
    return dataDF.join(NitrogenResponseRateDF, ['File'])\
        .withColumnRenamed('NResp', 'target_var')


def remove_sims_herbagecut_0(dataDF):
    '''Returns a pyspark.sql.dataframe.DataFrame
    
    dataDF: pyspark.sql.dataframe.DataFrame, a dataframe containing the weather and simulation data
    
    Removes the simulations where the herbage cut of the baseline 0
    '''
    
    return dataDF.filter(F.col('baseline_0_65_HerbageCut') != 0)


def remove_baseline_simulations(dataDF):
    '''Returns a pyspark.sql.dataframe.DataFrame
    
    dataDF: pyspark.sql.dataframe.DataFrame, a dataframe containing the weather and simulation data
    
    Removes the baseline simulations
    '''
    
    return dataDF.filter(F.col('FertRate') != 0)


def keep_relevant_days(dataDF, downlimit, uplimit):
    '''Returns a pyspark.sql.dataframe.DataFrame
    
    dataDF: pyspark.sql.dataframe.DataFrame, a dataframe containing the weather and simulation data
    downlimit: int, the earliest day (relative to fertilization) to be contained in our retained data
    uplimit: int, the latest day (relative to fertilization) to be contained in our retained data
    
    Retains only the rows of the days which are within the specified limits
    '''
    
    return dataDF.filter((F.col('DaysRelative') >= downlimit) &
                         (F.col('DaysRelative') <= uplimit))


def preprocessing_pipeline(climates, remove_herbagecut, downlimit, uplimit, biophysicals, simulations_path, weather_path, nitrogen_path):
    '''Returns None
    
    climates: which climates to process
    remove_herbagecut: bool, flag that shows if simulations which have in their corresponding baseline simulation herbage cut = 0 should be removed or not
    downlimit: int, earliest (inclusive) relative day before fertilization for which we assume to have data
    uplimit: int, latest (inclusive) relative day before fertilization for which we assume to have data
    biophysicals: list, biophysical variables of the simulations which we wish to retain
    simulations_path: str, the path to the csvs with the simulations
    weather_path: str, the path to the csvs with the weather
    nitrogen_path: str, the path to the cvs with the calculated nitrogen respose rates
    
    Orchestrates the preprocessing of the data and saves the results.
    '''
    
    # create a spark cluster
    spark = SparkSession.builder.master("local[25]")\
        .config('spark.driver.memory', '40g')\
        .config('spark.executor.memory', '8g')\
        .config('spark.network.timeout', '800s')\
        .config('spark.sql.legacy.timeParserPolicy', 'LEGACY')\
        .getOrCreate()

    # read data
    simulationsDF = spark.read.csv(
        simulations_path, header=True, mode='FAILFAST', schema=schemas.clover_simulations_schema)
    WeatherDataAllDF = spark.read.csv(
        weather_path, header=True, mode='FAILFAST', schema=schemas.weather_schema)
    NitrogenResponseRateDF = spark.read.csv(
        nitrogen_path, header=True, mode='FAILFAST', schema=schemas.nitrogen_response_rate_schema)

    NitrogenResponseRateDF = NitrogenResponseRateDF.filter(F.col(
        'PastureType') == 'GrassClover')  # keep only the values that refer to clover

    # setting up data before aggregations
    dataDF = combine_input_datasets(
        WeatherDataAllDF, simulationsDF, biophysicals)
        
    dataDF = extract_columns_from_filename(dataDF)
    dataDF = calculate_baseline_herbage_cuts(dataDF)
    dataDF = add_target_variable(dataDF, NitrogenResponseRateDF)
    if remove_herbagecut:
        dataDF = remove_sims_herbagecut_0(dataDF)
    dataDF = remove_baseline_simulations(dataDF)
    dataDF = keep_relevant_days(dataDF, downlimit, uplimit)

    dataDF.cache()
    
    
    #we have the year inside the following list because we need to preserve it to filter based on it for training test set
    sim_params = ['SoilWater', 'SoilFertility', 'Irrigation', 'FertMonth', 'FertRate', 'FertDay', 'Year']
    weather_biophysicals = ['Rain', 'MaxT', 'MinT', 'RH', 'VP', 'Wind', 'Radn', 'PET', 'AboveGroundWt', 'NetGrowthWt', 'NetPotentialGrowthWt', 'SoilWater300', 'SoilTemp300', 'SoilTemp050', 'AboveGroundNConc']
        
    agg_df = dataDF.groupby('File')\
            .pivot('DaysRelative', list(range(downlimit, uplimit + 1, 1)))\
            .agg(*[F.first(column_name).alias(column_name) for column_name in weather_biophysicals])
    
    agg_df.cache()
    
    
    agg_df = agg_df.select('File', *[F.col(column_name).alias(''.join(re.split('(_)', column_name)[::-1])) for column_name in agg_df.columns[1:]])
    
    
    agg_df = agg_df.select('File', *natsorted(agg_df.columns[1:])[::-1]) #the [::-1] is done to reverse the list of results because natsort doesn't interpret '-' as part of the number and sorts them like they were positive
        
    agg_df = agg_df.join(dataDF.select('File', 'Weather', 'target_var', *sim_params).distinct(), ['File']) #weather is the climate number, we need it to select climates later
    agg_df = agg_df.select('File', 'Weather', 'target_var', *agg_df.columns[1:-9], *agg_df.columns[-7:]) #-9 is 'Weather', -8 is 'target_var'
    
    validation_years = [1979, 1987, 1999, 2007]
    train_years = list(filter(lambda year: year not in validation_years, list(range(1979, 2011)))) 
    test_years = list(range(2011, 2019))
    train_agg_df = agg_df.filter(F.col('Year').isin(train_years))
    validation_agg_df = agg_df.filter(F.col('Year').isin(validation_years))
    test_agg_df = agg_df.filter(F.col('Year').isin(test_years))

    for climate in climates:
        if  climate != 'all':
            train_write_df = train_agg_df.filter(F.col('Weather') == climate)
            validation_write_df = validation_agg_df.filter(F.col('Weather') == climate)
            test_write_df = test_agg_df.filter(F.col('Weather') == climate)
                        
            train_write_df.write.csv('/mnt/guanabana/raid/home/pylia001/NZ/NNs/results/train_validation_test/' + climate + '/train.csv', header=True)
            validation_write_df.write.csv('/mnt/guanabana/raid/home/pylia001/NZ/NNs/results/train_validation_test/' + climate + '/validation.csv', header=True)
            test_write_df.write.csv('/mnt/guanabana/raid/home/pylia001/NZ/NNs/results/train_validation_test/' + climate + '/test.csv', header=True)   
        else:
            train_write_df.write.csv('/mnt/guanabana/raid/home/pylia001/NZ/NNs/results/train_validation_test/all_climates/train.csv', header=True)
            validation_write_df.write.csv('/mnt/guanabana/raid/home/pylia001/NZ/NNs/results/train_validation_test/all_climates/validation.csv', header=True)
            test_write_df.write.csv('/mnt/guanabana/raid/home/pylia001/NZ/NNs/results/train_validation_test/all_climates/test.csv', header=True) 

