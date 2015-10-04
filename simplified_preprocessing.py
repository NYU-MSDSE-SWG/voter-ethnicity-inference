import pandas as pd
import numpy as np
import random
import string
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib


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
        'name', 'perc', 'white', 'black', 'api', 'aian', '2race', 'latino']
    name_prob['other'] = 0
    other_list = ['aian', '2race']
    for other_race in other_list:
        name_prob.loc[:, 'other'] = name_prob.loc[
            :, 'other'] + name_prob.loc[:, other_race]
    race_list = ['white', 'black', 'api', 'other', 'latino']
    for race in race_list:
        name_prob[race] = name_prob[race] / float(100)
    name_prob['perc'] = name_prob['perc'] / float(100000)
    name_prob.rename(columns={'api': 'asian'}, inplace=True)
    name_prob.index = name_prob['name']
    name = name_prob.drop(['name', 'aian', '2race'], 1)
    return name


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
    elif x == 'latino':
        return 4
    elif x == 'other':
        return 6
    else:
        raise Exception('Undefined ethnic %s' % x)

def read_census(file_loc):
    census = pd.read_csv(file_loc)
    if (census.columns == [u'gisjoin', u'total', u'latino', u'white', u'black', u'asian', u'other']).all():
        float_type_list = ['total', 'latino', 'white', 'black', 'asian', 'other']
        census[float_type_list] = census[float_type_list].astype(float)

        normalize_list = ['latino', 'white', 'black', 'asian', 'other']
        for col in normalize_list:
            census[col] = census[col] / census['total']

        census['perc'] = census['total'] / census['total'].sum()
        census.index = census['gisjoin']
        census = census.drop('gisjoin', 1)
        return census
    else:
        raise Exception('Incorrect census file format.')



def create_location_ethnic_prob(cleaned_census_df, return_ethnic_perc=False):
    """
    Create a DataFrame containing location probability conditioned on ethnicity
    P(location | ethnicity)
    :param cleaned_census_df: DataFrame, from output of preprocess_census()
    :return location_ethnic_prob: DataFrame
            ethnic_perc: Series, containing percentage of each ethnicity
    """
    location_prob = cleaned_census_df[['total', 'white', 'black', 'asian', 'other',
                                       'latino', 'perc']]
    location_ethnic_prob = location_prob.copy()

    ethnic_list = ['white', 'black', 'asian', 'other', 'latino']
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


def read_voter(file_loc, sample=0, remove_name=False):
    voter = pd.read_csv(file_loc)
    column_match = np.sum(np.in1d(np.array(['voter_id', 'gisjoin10', 'lastname', 'firstname']),
                                  np.array(voter.columns)))
    if column_match == 4:
        voter = voter.dropna(axis=0)
        voter['lastname'] = voter['lastname'].map(lambda x: x.upper())
        voter['lastname'] = voter['lastname'].apply(string.strip)

        if remove_name:
            name_prob = preprocess_surname('./data/surname_list/app_c.csv')
            intlastname = np.in1d(voter['lastname'], name_prob.index)
            voter = voter[intlastname]

        if sample > 0:
            rows = random.sample(voter.index, sample)
            voter = voter.ix[rows]

        if 'race' in voter.columns:
            voter.race = voter.race.astype(float).astype(int)
            # map some race to 'other'
            voter.race = voter.race.replace({7: 6, 1: 6, 9: 6})

        return voter
    else:
        raise Exception('Wrong voter file format.')

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
    name_prob = name_prob[['white','black','asian','latino','other']]
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