import pandas as pd
import numpy as np
import random
import string
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib


def nc_race(x):
    if x.ethnic_code.strip() == 'HL':
        return 4
    else:
        if x.race_code.strip() == 'W':
            return 5
        elif x.race_code.strip() == 'B':
            return 3
        elif x.race_code.strip() == 'A':
            return 2
        else:
            return 6

def transform_output(x):
    """
    Tranform predict_ethnic output from ethnicity name to code in order to match voter's file
    :param x: string
    :return: int
    """
    if x == 'white':
        return 5
    elif x == 'black':
        return 3
    elif x == 'asian':
        return 2
    elif x == 'hispanic':
        return 4
    elif x == 'other':
        return 6
    else:
        raise Exception('Undefined ethnic %s' % x)


def preprocess_surname(file_loc):
    """
    This function preprocess surname list csv file and return a Dataframe
    containing ethnicity probability conditioned on surname.
    :param file_loc: string
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
        name_prob = pd.read_csv(file_loc)
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
    name = name_prob.drop(['name', 'aian', '2race'], 1)
    return name


def read_census(file_loc, census_type='group'):
    """
    This function read census file. If census data is block group level, it will extract
    ethnicty information. If census data is block level, it will just read csv file as
    DataFrame. This function is used in preprocess_census function.
    :param file: string, dataset file location
    :param census_type: string, specify what level census data is provided ('group' or 'block')
    :return: DataFrame
    For census block group data, it has column=['total', 'white', 'black', 'indian_alaska',
    'asian', 'hawaiian_islander', 'other', '2ormore', 'hispanic', 'TRACT', 'BLKGRP', 'COUNTY',
    'y5', 'y9', 'y14', 'y17', 'y24', 'y34', 'y44', 'y54', 'y64', 'y74', 'y84', 'y85o']]
    'total': total number of people in this census block group
    'white', 'black', 'indian_alaska', 'asian', 'hawaiian_islander', 'hispanic':
        number of people in this ethnic in this census block group
    '2ormore': number of people having two or more ethnicity
    'other': number of people having other ethnicity not described above
    'TRACT', 'BLKGRP', 'COUNTY': geographical code describing this census block group
    'y5', 'y9', 'y14', 'y17', 'y24', 'y34', 'y44', 'y54', 'y64', 'y74', 'y84':
        number of people within an interval of age in this census block group.
        eg: 'y9' represents number of people from age 5 to age 9
            'y14' represents number of people from age 10 to age 14
    'y85o': number of people 85 years and over
    """
    try:
        if census_type == 'group':
            census = pd.read_csv(file_loc, dtype=object)
            census = census[['SE_T015_001', 'SE_T015_003', 'SE_T015_004',
                             'SE_T015_005', 'SE_T015_006', 'SE_T015_007',
                             'SE_T015_008', 'SE_T015_009', 'SE_T015_010',
                             'Geo_TRACT', 'Geo_BLKGRP', 'Geo_COUNTY',
                             'SE_T008_002', 'SE_T008_003', 'SE_T008_004',
                             'SE_T008_005', 'SE_T008_006', 'SE_T008_007',
                             'SE_T008_008', 'SE_T008_009', 'SE_T008_010',
                             'SE_T008_011', 'SE_T008_012', 'SE_T008_013']]
        elif census_type == 'block':
            census = pd.read_csv(file_loc, dtype=object)
            return census
    except:
        raise Exception('Cannot open census csv file. Please make sure to '
                        'choose correct census_type, and download correct '
                        'census data. eg: census block group level from: '
                        'http://old.socialexplorer.com/pub/reportdata/'
                        'CsvResults.aspx?reportid=R10950075 '
                        'or census block level from NHGIS '
                        'https://www.nhgis.org')

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
    return census


def create_cbg2000(census_df, transform=False):
    """
    Create cbg2000 geocoding from borocode, tract and blkgrp in census block
    group data.
    :param census_df: DataFrame, cleaned census data output by read_census()
           transform: Boolean, transform=True when NY's census block group data
           and voter's file is used, else transform=False
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


def preprocess_census(file_loc, transform=False, census_type='group'):
    """
    Preprocess census data. It combines ethnicity information to percentage
    of people in white, black, asian, hispanic and other. It also gives
    percentage of people in different age range described in read_census().
    :param file_loc: string, location of census data
    :param transform: Boolean, transform=True when NY's census block group data
           and voter's file is used, else transform=False
    :param census_type: string, specify what level census data is provided
           ('group' or 'block')
    :return: DataFrame, having column=['white', 'black', 'asian', 'other',
             'hispanic', 'y5', 'y9', 'y14', 'y17', 'y24', 'y34', 'y44', 'y54',
             'y64', 'y74', 'y84', 'y85o', 'total'].
             Index is location geocode.
    """
    census = read_census(file_loc, census_type=census_type)
    if census_type == 'group':
        # For census block group data
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

        # combining indian_alaska and two more races to be other
        other_list = ['indian_alaska', '2race']
        for other_race in other_list:
            census.loc[:, 'other'] = census.loc[
                :, 'other'] + census.loc[:, other_race]

        census = census.drop(other_list, 1)
        census = census.drop(['TRACT', 'BLKGRP', 'COUNTY'], 1)

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

    elif census_type == 'block':
        # Census 2000
        if 'FX1001' in census.columns:
            census = census[['GISJOIN', 'FX1001', 'FX1002', 'FX1003', 'FX1004','FX1005','FX1006','FXZ001']]
            census[['FX1001', 'FX1002', 'FX1003', 'FX1004','FX1005','FX1006','FXZ001']] = \
                census[['FX1001', 'FX1002', 'FX1003', 'FX1004','FX1005','FX1006','FXZ001']].astype(float)
            census['total'] = census.sum(axis=1)
            col_dict = {'FX1001': 'white', 'FX1002': 'black', 'FX1003': 'indian_alaska',
                        'FX1004': 'asian', 'FX1005': 'hawaiian_islander', 'FX1006': 'other',
                        'FXZ001': 'hispanic'}
        # Census 2010
        elif 'H7Z010' in census.columns:
            census = census[['GISJOIN', 'H7Z003', 'H7Z004', 'H7Z005', 'H7Z006','H7Z007','H7Z008','H7Z010']]
            census[['H7Z003', 'H7Z004', 'H7Z005', 'H7Z006','H7Z007','H7Z008','H7Z010']] = \
                census[['H7Z003', 'H7Z004', 'H7Z005', 'H7Z006','H7Z007','H7Z008','H7Z010']].astype(float)
            census['total'] = census.sum(axis=1)
            col_dict = {'H7Z003': 'white', 'H7Z004': 'black', 'H7Z005': 'indian_alaska',
                        'H7Z006': 'asian', 'H7Z007': 'hawaiian_islander', 'H7Z008': 'other',
                        'H7Z010': 'hispanic'}
        else:
            raise Exception('Unknown census file')

        census.rename(columns=col_dict, inplace=True)
        census.loc[:, 'asian'] = census.loc[
            :, 'asian'] + census.loc[:, 'hawaiian_islander']
        other_list = ['indian_alaska']
        for other_race in other_list:
            census.loc[:, 'other'] = census.loc[
                :, 'other'] + census.loc[:, other_race]
        normalize_list = ['white', 'black', 'asian', 'other', 'hispanic']
        for col in normalize_list:
            census[col] = census[col] / census['total']
        census['perc'] = census['total'] / census['total'].sum()
        census.index = census['GISJOIN']
        return census
    else:
        raise Exception('Undefined census type %s' %census_type)

def create_location_prob(cleaned_census_df):
    """
    Extract ethnicity probability conditioned on location from cleaned census DataFrame
    :param cleaned_census_df: DataFrame, output from preprocess_census()
    :return: DataFrame, having column=['total', 'white', 'black', 'asian', 'other',
            'hispanic', 'perc'].
            Index is location geocode.
    """
    census = cleaned_census_df
    # create a dataframe containing ethnicity probability conditioned on block
    # location
    location_prob = census[
        ['total', 'white', 'black', 'asian', 'other', 'hispanic', 'perc']]
    return location_prob


def create_age_prob(cleaned_census_df):
    """
    Extract age probability conditioned on location from cleaned census DataFrame
    :param cleaned_census_df: DataFrame, output from preprocess_census()
    :return: DataFrame, having column=['total', 'y5', 'y9', 'y14', 'y17', 'y24',
                       'y34', 'y44', 'y54', 'y64', 'y74', 'y84', 'y85o', 'perc']
    """
    census = cleaned_census_df
    # create a dataframe containing age probability conditioned on block
    # location
    age_prob = census[['total', 'y5', 'y9', 'y14', 'y17', 'y24',
                       'y34', 'y44', 'y54', 'y64', 'y74', 'y84', 'y85o', 'perc']]
    return age_prob


def create_location_ethnic_prob(cleaned_census_df, return_ethnic_perc=False):
    """
    Create a DataFrame containing location probability conditioned on ethnicity
    P(location | ethnicity)
    :param cleaned_census_df: DataFrame, from output of preprocess_census()
    :return location_ethnic_prob: DataFrame
            ethnic_perc: Series, containing percentage of each ethnicity
    """
    location_prob = cleaned_census_df[['total', 'white', 'black', 'asian', 'other',
                                       'hispanic', 'perc']]
    location_ethnic_prob = location_prob.copy()

    ethnic_list = ['white', 'black', 'asian', 'other', 'hispanic']
    ethnic_perc = dict()
    for ethnic in ethnic_list:
        temp = location_prob[ethnic] * location_prob['perc']
        ethnic_perc[ethnic] = temp.sum()
        location_ethnic_prob.loc[:, ethnic] = temp / ethnic_perc[ethnic]
    ethnic_perc = pd.Series(ethnic_perc)
    if return_ethnic_perc:
        return location_ethnic_prob, ethnic_perc
    else:
        return location_ethnic_prob


def validate_input(lastname, cbg2000):
    """
    Check whether lastname and cbg2000 have same length.
    :param lastname: string or list
    :param cbg2000: string or list
    :return: lastname_list: list
             cbg2000_list: list
    """
    lastname_list = lastname
    cbg2000_list = cbg2000
    if isinstance(lastname, str):
        lastname_list = [lastname]
    if isinstance(cbg2000, str):
        cbg2000_list = [cbg2000]
    if len(cbg2000_list) != len(lastname_list) and len(cbg2000) > 0:
        raise Exception(
            'Input lastname list and cbg2000 list should have same length')
    return lastname_list, cbg2000_list


def read_voter(file_loc):
    """
    Read voter's file and return a DataFrame
    :param file_loc: string, location of voter's file
    :return: DataFrame
    """
    if isinstance(file_loc, str):
        file_type = file_loc.split('.')[-1]
        if file_type == 'csv':
            voter_file = pd.read_csv(file_loc, dtype=object)
        elif file_type == 'dta':
            voter_file = pd.read_stata(file_loc, preserve_dtypes=False,
                                       convert_categoricals=False, convert_dates=False)
        else:
            raise Exception("Can not open voter's file, please input a csv or dta file")
        return voter_file
    else:
        raise Exception("Please input string as file location")


def preprocess_voter(file_loc, census_type='group', sample=0, remove_name=True):
    """
    Preprocess voter's file. It will drop rows with na in ['voter_id', 'gisjoin10',
    'gisjoin00', 'lastname', 'firstname', 'gender', 'race', 'birth_date']. If
    type='group', it will also create geocode from county, tract and blkgroup to match
    census block group file.
    :param file_loc: string
    :param type: string, 'group' or 'block'
    :param sample: int, if greater than 0, it will sample rows from voter file
    :param remove_name: boolean, if True, it will remove voter whose surname is not in
           census name list.
    :return:
    """
    test = read_voter(file_loc)
    print("Finish reading from file")
    if sample > 0:
        rows = random.sample(test.index, sample)
        test = test.ix[rows]

    if census_type == 'group':
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
    elif census_type == 'block':
        try:
            id_use = ['voter_id', 'gisjoin10', 'gisjoin00', 'lastname',
                      'firstname', 'gender', 'race', 'birth_date']
            test = test[id_use]
            test = test.dropna(axis=0)
        except:
            id_use = ['voter_reg_num', 'gisjoin10', 'gisjoin00',
                      'last_name', 'first_name', 'sex_code', 'ethnic_code', 'race_code']
            test = test[id_use]
            test = test.dropna(axis=0)

        col_dict = {'voter_reg_num': 'voter_id', 'last_name': 'lastname',
                    'first_name': 'firstname'}
        test.rename(columns=col_dict, inplace=True)

    else:
        raise Exception('Undefined type %s' % census_type)

    # remove rows having lastname not in census name list

    if 'ethnic_code' in test.columns:
        print("Starting applying race")
        form_race = test.apply(nc_race, axis=1)
        test['race'] = form_race

    test.race = test.race.astype(float).astype(int)
    test['lastname'] = test['lastname'].map(lambda x: x.upper())
    test['lastname'] = test['lastname'].apply(string.strip)
    if remove_name == True:
        name_prob = preprocess_surname('./data/surname_list/app_c.csv')
        intlastname = np.in1d(test['lastname'], name_prob.index)
        test = test[intlastname]

    # combine some ethnics to 'other'
    test.race = test.race.replace({7: 6, 1: 6, 9: 6})
    return test


def create_name_predictor(file_loc, n_gram=(2,5), save=True):
    """
    Using character level n_gram and logistic regression to train a classification
    model to predict ethnicity based on surname only.
    :param file_loc: string, surname list file location
    :param n_gram: tuple (min_n, max_n), The lower and upper boundary of the range
    of n-values for different n-grams to be extracted. All values of n such that
    min_n <= n <= max_n will be used.
    :param save: boolean, if True, it will save models to ./model/ directory
    :return: n_gram_model to vectorize string and classifier to do classification
    """
    name_prob = preprocess_surname(file_loc).fillna(0)
    name_prob = name_prob[pd.Series(name_prob.index).notnull().tolist()]
    name_prob = name_prob[['white','black','asian','hispanic','other']]
    name_prob['label'] = name_prob.idxmax(axis=1)
    name_list = pd.Series(name_prob.index.tolist())
    n_gram_model = CountVectorizer(analyzer='char', ngram_range=n_gram)
    train_x = n_gram_model.fit_transform(name_list.tolist())
    classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    classifier.fit(train_x, name_prob['label'])
    if save == True:
        if not os.path.exists('./model/'):
            os.makedirs('./model/')
        joblib.dump(n_gram_model, './model/n_gram.pkl')
        joblib.dump(classifier, './model/classifier.pkl')
    return n_gram_model, classifier


def n_gram_name_prob(n_gram_model, classifier, surname):
    """
    Create surname_ethnicity probability dataframe using n_gram_model and classifier.
    :param n_gram_model: saved n_gram_model or returned by create_name_predictor
    :param classifier: saved classifier_model or returned by create_name_predictor
    :param surname: list, list of surname
    :return: DataFrame, containing surname_ethnicity probability
    """
    name_col = classifier.classes_
    test_x = n_gram_model.transform(surname)
    predict_prob = classifier.predict_proba(test_x)
    return pd.DataFrame(predict_prob, columns=name_col, index=surname)


if __name__ == '__main__':
    name = preprocess_surname('./data/surname_list/app_c.csv')
    print(name.iloc[:3])
    census = preprocess_census('./data/Census2000_BG/C2000_NY.csv', transform=True, census_type='group')
    location_ethnic_prob = create_location_ethnic_prob(census)
    print(location_ethnic_prob.iloc[:3])
