import pandas as pd


def preprocess_surname(file):
    """
    This function preprocess surname list csv file and return a Dataframe containing ethnicity probability conditioned
    on surname.
    :param file: string
    :return: DataFrame, having column=['name', 'perc', 'white', 'black', 'api', 'aian', '2race', 'hispanic']

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
    name_prob = name_prob[[u'name', u'prop100k', u'pctwhite', u'pctblack', u'pctapi', u'pctaian', u'pct2prace', u'pcthispanic']]
    name_prob.columns = ['name', 'perc', 'white', 'black', 'api', 'aian', '2race', 'hispanic']
    race_list = ['white', 'black', 'api', 'aian', '2race', 'hispanic']
    for race in race_list:
        name_prob[race] = name_prob[race] / 100
    name_prob['perc'] = name_prob['perc'] / 100000
    name_prob.index = name_prob['name']
    name = name_prob.drop('name', 1)
    return name


def read_census(file):
    try:
        census = pd.read_csv(file, dtype=object)
        census = census[['SE_T015_001','SE_T015_003','SE_T015_004','SE_T015_005','SE_T015_006','SE_T015_007','SE_T015_008',
             'SE_T015_009','SE_T015_010','Geo_TRACT','Geo_BLKGRP','Geo_COUNTY','SE_T008_002','SE_T008_003','SE_T008_004',
             'SE_T008_005','SE_T008_006','SE_T008_007','SE_T008_008','SE_T008_009','SE_T008_010','SE_T008_011',
             'SE_T008_012','SE_T008_013']]
    except:
        raise Exception('Cannot open census csv file. Please download correct block group data. eg: http://old.socialexplorer.com/pub/reportdata/CsvResults.aspx?reportid=R10950075')

    col_dict = {'SE_T015_001': 'total', 'SE_T015_003': 'white', 'SE_T015_004': 'black', 'SE_T015_005': 'indian_alaska',
                'SE_T015_006': 'asian', 'SE_T015_007': 'hawaiian_islander', 'SE_T015_008': 'other', 'SE_T015_009': '2race',
                'SE_T015_010': 'hispanic', 'Geo_TRACT': 'TRACT', 'Geo_BLKGRP': 'BLKGRP', 'Geo_COUNTY': 'COUNTY',
                'SE_T008_002': 'y5', 'SE_T008_003': 'y9', 'SE_T008_004': 'y14', 'SE_T008_005': 'y17', 'SE_T008_006': 'y24',
                'SE_T008_007': 'y34', 'SE_T008_008': 'y44', 'SE_T008_009': 'y54', 'SE_T008_010': 'y64', 'SE_T008_011': 'y74',
                'SE_T008_012': 'y84', 'SE_T008_013': 'y85o'
                }

    census.rename(columns=col_dict, inplace=True)
    #census.columns = ['total','white','black','indian_alaska','asian','hawaiian_islander','other','2ormore','hispanic',
                  #'TRACT','BLKGRP','COUNTY','y5','y9','y14','y17','y24','y34','y44','y54','y64','y74','y84','y85o']
    return census


def create_cbg2000(census_df):
    """
    Create cbg2000 geocoding from borocode, tract and blkgrp in census data
    :param census_df: DataFrame, cleaned census data output by read_census()
    :return: DataFrame, having a new column called 'cbg2000'
    """
    census = census_df

    # creating cbg2000 geocode
    census['borocode'] = ''
    census.loc[census['COUNTY'] == '047', 'borocode'] = '3'
    census.loc[census['COUNTY'] == '081', 'borocode'] = '4'
    census.loc[census['COUNTY'] == '061', 'borocode'] = '1'
    census.loc[census['COUNTY'] == '005', 'borocode'] = '2'
    census.loc[census['COUNTY'] == '085', 'borocode'] = '5'
    census['cbg2000'] = ''
    census.loc[:, 'cbg2000'] = census.loc[:, 'borocode'] + census.loc[:, 'TRACT'] + census.loc[:, 'BLKGRP']

    census = census.drop('borocode',1)

    return census


def preprocess_census(file):
    census = read_census(file)
    census = create_cbg2000(census)


    float_type_list = ['total','white','black','indian_alaska','asian','hawaiian_islander','other','2race','hispanic',
                       'y5','y9','y14','y17','y24','y34','y44','y54','y64','y74','y84','y85o']
    census[float_type_list] = census[float_type_list].astype(float)

    # combining asian and hawaiian islander to be asian and pacific islander (aian)
    census.loc[:,'asian'] = census.loc[:,'asian'] + census.loc[:,'hawaiian_islander']

    census = census.drop('hawaiian_islander',1)
    census = census.drop('other',1)
    census = census.drop(['TRACT','BLKGRP','COUNTY'],1)
    census.rename(columns={'indian_alaska': 'aian','asian':'api'}, inplace=True)

    # normalize count to percentage
    normalize_list = ['white','black','aian','api','2race','hispanic',
                       'y5','y9','y14','y17','y24','y34','y44','y54','y64','y74','y84','y85o']
    for col in normalize_list:
        census[col] = census[col]/census['total']
    census['perc'] = census['total']/census['total'].sum()
    census.index = census['cbg2000']
    census = census.drop('cbg2000', 1)
    return census


def create_location_prob(cleaned_census_df):
    census = cleaned_census_df
    # create a dataframe containing ethnicity probability conditioned on block location
    location_prob = census[['total','white','black','api','aian','2race','hispanic','perc']]
    return location_prob


def create_age_prob(cleaned_census_df):
    census = cleaned_census_df
    # create a dataframe containing age probability conditioned on block location
    age_prob = census[['total','y5','y9','y14','y17','y24','y34','y44','y54','y64','y74','y84','y85o','perc']]
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

    ethnic_list = ['white','black','api','aian','2race','hispanic']
    ethnic_perc = dict()
    for ethnic in ethnic_list:
        ethnic_perc[ethnic] = (location_prob[ethnic] * location_prob['perc']).sum()
        location_ethnic_prob.loc[:, ethnic] = location_prob.loc[:, ethnic] * location_prob.loc[:, 'perc'] / ethnic_perc[ethnic]
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
        raise Exception('Input lastname list and cbg2000 list should have same length')
    return lastname_list, cbg2000_list


if __name__ == '__main__':
    name = preprocess_surname('./data/surname_list/app_c.csv')
    print(name.iloc[:3])
    census = preprocess_census('./data/Census2000_BG/C2000_NY.csv')
    location_prob = create_location_prob(census)
    location_ethnic_prob = create_location_ethnic_prob(location_prob)
    print(location_ethnic_prob.iloc[:3])