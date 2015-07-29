import pandas as pd
import numpy as np


def preprocess_surname(file):
    """
    This function preprocess surname list csv file and return a Dataframe
    containing ethnicity probability conditioned on surname.
    :param file: string
    :return: DataFrame, having column=['name', 'perc', 'white', 'black', 'api',
    'aian', '2race', 'hispanic']

    'perc' : percentage of this name in total population
    'white' : Percent Non-Hispanic White
    'black' : Percent Non-Hispanic Black
    'api' : Percent Non-Hispanic Asian and Pacific Islander
    'aian' : Percent Non-Hispanic American Indian and Alaskan Native
    '2race' : Percent Non-Hispanic of Two or More Races
    'hispanic' : Percent Hispanic Origin
    """
    try:
        name_prob = pd.read_csv(file)
    except:
        raise Exception('Cannot open surname list csv file')
    name_prob = name_prob.convert_objects(convert_numeric=True)
    name_prob = name_prob[[u'name', u'prop100k', u'pctwhite',
                           u'pctblack', u'pctapi', u'pctaian', u'pct2prace', u'pcthispanic']]
    name_prob.columns = [
        'name', 'perc', 'white', 'black', 'api', 'aian', '2race', 'hispanic']
    name_prob['other'] = 0
    other_list = ['aian', '2race']
    for other_race in other_list:
        name_prob.loc[:, 'other'] = name_prob.loc[
            :, 'other'] + name_prob.loc[:, other_race]
    race_list = ['white', 'black', 'api', 'other', 'hispanic']
    for race in race_list:
        name_prob[race] = name_prob[race] / float(100)
    name_prob['perc'] = name_prob['perc'] / float(100000)
    name_prob.rename(columns={'api': 'asian'}, inplace=True)
    name_prob.index = name_prob['name']
    name = name_prob.drop('name', 1)
    return name


def read_census(file, census_type='group'):
    """
    This function read census file, and extract ethnicty information for
    census block level dataset
    :param file: string, dataset file location
    :param census_type: string, specify what level census data is provided
    :return: DataFrame
    """
    try:
        if census_type == 'group':
            census = pd.read_csv(file, dtype=object)
            census = census[['SE_T015_001', 'SE_T015_003', 'SE_T015_004',
                             'SE_T015_005', 'SE_T015_006', 'SE_T015_007',
                             'SE_T015_008', 'SE_T015_009', 'SE_T015_010',
                             'Geo_TRACT', 'Geo_BLKGRP', 'Geo_COUNTY',
                             'SE_T008_002', 'SE_T008_003', 'SE_T008_004',
                             'SE_T008_005', 'SE_T008_006', 'SE_T008_007',
                             'SE_T008_008', 'SE_T008_009', 'SE_T008_010',
                             'SE_T008_011', 'SE_T008_012', 'SE_T008_013']]
        elif census_type == 'block':
            census = pd.read_csv(file, dtype=object)
            return census
    except:
        raise Exception('Cannot open census csv file. Please download correct'
                        'block group data. eg: '
                        'http://old.socialexplorer.com/pub/reportdata/'
                        'CsvResults.aspx?reportid=R10950075')

    col_dict = {'SE_T015_001': 'total', 'SE_T015_003': 'white', 'SE_T015_004':
                'black', 'SE_T015_005': 'indian_alaska', 'SE_T015_006':
                'asian', 'SE_T015_007': 'hawaiian_islander', 'SE_T015_008':
                'other', 'SE_T015_009': '2race', 'SE_T015_010': 'hispanic',
                'Geo_TRACT': 'TRACT', 'Geo_BLKGRP': 'BLKGRP', 'Geo_COUNTY':
                'COUNTY', 'SE_T008_002': 'y5', 'SE_T008_003': 'y9',
                'SE_T008_004': 'y14', 'SE_T008_005': 'y17', 'SE_T008_006':
                'y24', 'SE_T008_007': 'y34', 'SE_T008_008': 'y44',
                'SE_T008_009': 'y54', 'SE_T008_010': 'y64', 'SE_T008_011':
                'y74', 'SE_T008_012': 'y84', 'SE_T008_013': 'y85o'}

    census.rename(columns=col_dict, inplace=True)
    #census.columns = ['total', 'white', 'black', 'indian_alaska', 'asian',
    #                  'hawaiian_islander', 'other', '2ormore', 'hispanic',
    #                  'TRACT', 'BLKGRP', 'COUNTY', 'y5', 'y9', 'y14', 'y17',
    #                  'y24', 'y34', 'y44', 'y54', 'y64', 'y74', 'y84', 'y85o']
    return census


def create_cbg2000(census_df, transform=True):
    """
    Create cbg2000 geocoding from borocode, tract and blkgrp in census data
    :param census_df: DataFrame, cleaned census data output by read_census()
    :return: DataFrame, having a new column called 'cbg2000'
    """
    census = census_df

    # creating cbg2000 geocode
    if transform is True:
        census['borocode'] = ''
        census.loc[census['COUNTY'] == '047', 'borocode'] = '3'
        census.loc[census['COUNTY'] == '081', 'borocode'] = '4'
        census.loc[census['COUNTY'] == '061', 'borocode'] = '1'
        census.loc[census['COUNTY'] == '005', 'borocode'] = '2'
        census.loc[census['COUNTY'] == '085', 'borocode'] = '5'
        census['cbg2000'] = ''
        census.loc[:, 'cbg2000'] = census.loc[:, 'borocode'] + \
            census.loc[:, 'TRACT'] + census.loc[:, 'BLKGRP']

        census = census.drop('borocode', 1)
    else:
        census.loc[:, 'cbg2000'] = census.loc[:, 'COUNTY'] + \
            census.loc[:, 'TRACT'] + census.loc[:, 'BLKGRP']

    return census


def preprocess_census(file, transform=True, census='group'):
    census = read_census(file, census_type=census)
    if census.columns[0] == 'total':
        # For block group data
        census = create_cbg2000(census, transform)

        float_type_list = ['total', 'white', 'black', 'indian_alaska', 'asian',
                           'hawaiian_islander', 'other', '2race', 'hispanic',
                           'y5', 'y9', 'y14', 'y17', 'y24', 'y34', 'y44',
                           'y54', 'y64', 'y74', 'y84', 'y85o']
        census[float_type_list] = census[float_type_list].astype(float)

        # combining asian and hawaiian islander to be asian and pacific
        # islander (aian)
        census.loc[:, 'asian'] = census.loc[
            :, 'asian'] + census.loc[:, 'hawaiian_islander']
        other_list = ['hawaiian_islander', 'indian_alaska', '2race']
        for other_race in other_list:
            census.loc[:, 'other'] = census.loc[
                :, 'other'] + census.loc[:, other_race]

        census = census.drop(other_list, 1)
        census = census.drop(['TRACT', 'BLKGRP', 'COUNTY'], 1)
        #census.rename(columns={'indian_alaska': 'aian','asian':'api'}, inplace=True)

        # normalize count to percentage
        normalize_list = ['white', 'black', 'asian', 'other', 'hispanic', 'y5',
                          'y9', 'y14', 'y17', 'y24', 'y34', 'y44', 'y54',
                          'y64', 'y74', 'y84', 'y85o']
        for col in normalize_list:
            census[col] = census[col] / census['total']
        census['perc'] = census['total'] / census['total'].sum()
        census.index = census['cbg2000']
        census = census.drop('cbg2000', 1)
        return census
    else:
        census = census[['GISJOIN', 'FX1001', 'FX1002', 'FX1003', 'FX1004','FX1005','FX1006','FXZ001']]
        census[['FX1001', 'FX1002', 'FX1003', 'FX1004','FX1005','FX1006','FXZ001']] = \
            census[['FX1001', 'FX1002', 'FX1003', 'FX1004','FX1005','FX1006','FXZ001']].astype(float)
        census['total'] = census.sum(axis=1)
        col_dict = {'FX1001': 'white', 'FX1002': 'black', 'FX1003': 'indian_alaska',
                    'FX1004': 'asian', 'FX1005': 'hawaiian_islander', 'FX1006': 'other',
                    'FXZ001': 'hispanic'}
        census.rename(columns=col_dict, inplace=True)
        census.loc[:, 'asian'] = census.loc[
            :, 'asian'] + census.loc[:, 'hawaiian_islander']
        other_list = ['hawaiian_islander', 'indian_alaska']
        for other_race in other_list:
            census.loc[:, 'other'] = census.loc[
                :, 'other'] + census.loc[:, other_race]
        normalize_list = ['white', 'black', 'asian', 'other', 'hispanic']
        for col in normalize_list:
            census[col] = census[col] / census['total']
        census['perc'] = census['total'] / census['total'].sum()
        census.index = census['GISJOIN']
        return census

def create_location_prob(cleaned_census_df):
    census = cleaned_census_df
    # create a dataframe containing ethnicity probability conditioned on block
    # location
    location_prob = census[
        ['total', 'white', 'black', 'asian', 'other', 'hispanic', 'perc']]
    return location_prob


def create_age_prob(cleaned_census_df):
    census = cleaned_census_df
    # create a dataframe containing age probability conditioned on block
    # location
    age_prob = census[['total', 'y5', 'y9', 'y14', 'y17', 'y24',
                       'y34', 'y44', 'y54', 'y64', 'y74', 'y84', 'y85o', 'perc']]
    return age_prob


def create_location_ethnic_prob(location_prob_df, return_ethnic_perc=False):
    """
    Create a DataFrame containing block probability conditioned on ethnicity
    :param location_prob_df: DataFrame, from output of create_location_prob()
    :return location_ethnic_prob: DataFrame
            ethnic_perc: Series, containing percentage of each ethnicity
    """
    location_prob = location_prob_df
    location_ethnic_prob = location_prob.copy()

    ethnic_list = ['white', 'black', 'asian', 'other', 'hispanic']
    ethnic_perc = dict()
    for ethnic in ethnic_list:
        ethnic_perc[ethnic] = (
            location_prob[ethnic] * location_prob['perc']).sum()
        location_ethnic_prob.loc[:, ethnic] = location_prob.loc[
            :, ethnic] * location_prob.loc[:, 'perc'] / ethnic_perc[ethnic]
    ethnic_perc = pd.Series(ethnic_perc)
    if return_ethnic_perc:
        return location_ethnic_prob, ethnic_perc
    else:
        return location_ethnic_prob


def validate_input(lastname, cbg2000):
    lastname_list = lastname
    cbg2000_list = cbg2000
    if isinstance(lastname, str):
        lastname_list = [lastname]
    if isinstance(cbg2000, str):
        cbg2000_list = [cbg2000]
    if len(cbg2000_list) != len(lastname_list):
        raise Exception(
            'Input lastname list and cbg2000 list should have same length')
    return lastname_list, cbg2000_list


def preprocess_voter(test, type='group'):
    if type == 'group':
        id_use = ['voter_id', 'county', 'tract', 'blkgroup', 'lastname',
                  'firstname', 'gender', 'race', 'birth_date', '_merge']
        str_use = ['county', 'tract', 'blkgroup']
        test = test[id_use]
        test = test.dropna(axis=0)
        test[str_use] = test[str_use].astype(int).astype(str)
        test['county'] = test['county'].map(lambda x: x.rjust(3, '0'))
        test['tract'] = test['tract'].map(lambda x: x.rjust(6, '0'))
        test.loc[:, 'bctcb2000'] = test.loc[:, 'county'] + \
            test.loc[:, 'tract'] + test.loc[:, 'blkgroup']
        test['lastname'] = test['lastname'].map(lambda x: x.upper())
        name_prob = preprocess_surname('./data/surname_list/app_c.csv')
        intlastname = np.in1d(test['lastname'], name_prob.index)
        test = test[intlastname]
        test.race = test.race.replace({7: 6, 1: 6, 9: 6})
        return test
    elif type == 'block':
        id_use = ['voter_id', 'gisjoin10', 'gisjoin00', 'lastname',
                  'firstname', 'gender', 'race', 'birth_date']
        test.race = test.race.astype(float).astype(int)
        test = test[id_use]
        test = test.dropna(axis=0)
        test.loc[:, 'lastname'] = test.lastname.apply(lambda x: x.upper())
        name_prob = preprocess_surname('./data/surname_list/app_c.csv')
        intlastname = np.in1d(test['lastname'], name_prob.index)
        test.race = test.race.replace({7: 6, 1: 6, 9: 6})
        return test

if __name__ == '__main__':
    name = preprocess_surname('./data/surname_list/app_c.csv')
    print(name.iloc[:3])
    census = preprocess_census('./data/Census2000_BG/C2000_NY.csv')
    location_prob = create_location_prob(census)
    location_ethnic_prob = create_location_ethnic_prob(location_prob)
    print(location_ethnic_prob.iloc[:3])
