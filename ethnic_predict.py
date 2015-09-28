import pandas as pd
import numpy as np
import os
from sklearn.externals import joblib

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc

from simplified_preprocessing import (validate_input, preprocess_surname,
                           read_census, read_voter,
                           create_location_ethnic_prob,
                           transform_output, create_name_predictor)


def predict_ethnic(lastname, cbg2000, name_prob, location_ethnic_prob, verbose_prob=False, verbose_all=False):
    """
    Predicting ethnicity given surname(lastname), location code(cbg2000),
    name probability(name_prob), location probability conditioned on ethnicity(location_ethnic_prob)
    :param lastname: list, containing people's lastname
    :param cbg2000: list, containing people's geolocation code
    :param name_prob: DataFrame, output from preprocess_surname()
    :param location_ethnic_prob: DataFrame, output from create_location_ethnic_prob()
    :param verbose_prob: Boolean, if True, will return probability for predicted ethnic
    :param verbose_all: Boolean, if True and verbose_prob is True, will return probability for all ethnic
            (useful when drawing roc curve)
    :return: ethnic_pred_race: list, containing predicted ethnic
             ethnic_pred_prob: list, containing probability of predicted ethnic
    """
    lastname, cbg2000 = validate_input(lastname, cbg2000)
    name_p = name_prob.loc[lastname][
        ['white', 'black', 'asian', 'other', 'latino']]
    if len(cbg2000) != 0:
        location_ethnic_p = location_ethnic_prob.loc[cbg2000][
            ['white', 'black', 'asian', 'other', 'latino']]
        name_p = name_p.reset_index().drop('name', axis=1)
        location_ethnic_p = location_ethnic_p.reset_index()
        location_ethnic_p = location_ethnic_p[['white', 'black', 'asian', 'other', 'latino']]
        numerator = location_ethnic_p * name_p
        denominator = numerator.sum(axis=1)
        ans = numerator.div(denominator, axis='index').fillna(0)

        ethnic_pred_race = ans.idxmax(axis=1).tolist()
        ethnic_pred_prob = ans.max(axis=1).tolist()
    else:
        ethnic_pred_race = name_p.idxmax(axis=1).tolist()
        ethnic_pred_prob = name_p.max(axis=1).tolist()
        ans = name_p.fillna(0)
    if verbose_prob:
        if verbose_all:
            return ethnic_pred_race, ans
        else:
            return ethnic_pred_race, ethnic_pred_prob
    else:
        return ethnic_pred_race


def voter_file_predict(voter_loc, census_loc, namelist_loc, remove_name=False, sample=0):
    name_prob = preprocess_surname(namelist_loc)
    census = read_census(census_loc)
    voter = read_voter(voter_loc, sample=sample, remove_name=remove_name)
    file_name = os.path.split(voter_loc)[1]
    print('Predicting on %s' % file_name)
    location_ethnic_prob, ethnic_perc = create_location_ethnic_prob(census, True)

    if not remove_name:
        print('USE N-GRAM TO PREDICT VOTER NOT ON THE NAME LIST')
        notinlistname = np.setdiff1d(voter['lastname'], name_prob.index)
        try:
            n_gram_model = joblib.load('./model/n_gram.pkl')
            classifier = joblib.load('./model/classifier.pkl')
        except:
            n_gram_model, classifier = create_name_predictor('./data/surname_list/app_c.csv')
        notinname_prob = classifier.predict_proba(n_gram_model.transform(notinlistname))
        notinname_df = pd.DataFrame(notinname_prob, columns=classifier.classes_,index=notinlistname)
        notinname_df.index.name = 'name'
        name_prob = name_prob.append(notinname_df)

    print('Sample size %d' % len(voter))
    print('START PREDICTING')
    surname = voter['lastname']
    cbg2000 = voter['gisjoin10']
    predict, predict_ethnic_prob = predict_ethnic(
        surname, cbg2000, name_prob, location_ethnic_prob, True, True)
    voter = voter.reset_index()
    voter = voter.drop('index', 1)
    voter[['white', 'black', 'asian', 'other', 'latino']] = predict_ethnic_prob
    predict = pd.Series(predict).apply(transform_output)
    voter['predict_race'] = predict
    voter.to_csv('./voter_file_predicted.csv')
    if 'race' in voter.columns:
        print('Accuracy: %f' % accuracy_score(predict, voter['race']))
        print(classification_report(predict, voter['race']))
        print(confusion_matrix(predict, voter['race']))
    print('FINISH')
    return voter


def main():
    namelist_loc = './data/surname_list/app_c.csv'
    census_loc = './data/census/CensusBLK2010_FL_new.csv'
    voter_loc = './data/voter/FL_voterfile.csv'

    ans = voter_file_predict(voter_loc=voter_loc, census_loc=census_loc, namelist_loc=namelist_loc,
                             remove_name=False, sample=0)


if __name__ == '__main__':
    main()
