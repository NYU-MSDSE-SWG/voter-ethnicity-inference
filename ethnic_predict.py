import random
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from preprocessing import (validate_input, preprocess_surname,
                           preprocess_census, preprocess_voter,
                           create_location_ethnic_prob)


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


def predict_ethnic(lastname, cbg2000, name_prob, location_ethnic_prob, verbose_prob=False):
    """
    Predicting ethnicity given surname(lastname), location code(cbg2000),
    name probability(name_prob), location probability conditioned on ethnicity(location_ethnic_prob)
    :param lastname: list, containing people's lastname
    :param cbg2000: list, containing people's geolocation code
    :param name_prob: DataFrame, output from preprocess_surname()
    :param location_ethnic_prob: DataFrame, output from create_location_ethnic_prob()
    :param verbose_prob: Boolean, if True, will return probability for predicted ethnic
    :return: ethnic_pred_race: list, containing predicted ethnic
             ethnic_pred_prob: list, containing probability of predicted ethnic
    """
    lastname, cbg2000 = validate_input(lastname, cbg2000)
    name_p = name_prob.loc[lastname][
        ['white', 'black', 'asian', 'other', 'hispanic']]
    location_ethnic_p = location_ethnic_prob.loc[cbg2000][
        ['white', 'black', 'asian', 'other', 'hispanic']]
    name_p = name_p.reset_index().drop('name', axis=1)
    location_ethnic_p = location_ethnic_p.reset_index().drop('GISJOIN', axis=1)
    numerator = location_ethnic_p * name_p
    denominator = numerator.sum(axis=1)
    ans = numerator.div(denominator, axis='index').fillna(0)

    ethnic_pred_race = ans.idxmax(axis=1).tolist()
    ethnic_pred_prob = ans.max(axis=1).tolist()

    if verbose_prob:
        return ethnic_pred_race, ethnic_pred_prob
    else:
        return ethnic_pred_race


def main():
    """
    Usage: 1. Use preprocess_surname() to get name_prob
           2. Use preprocess_census() to get cleaned census
           3. Use create_location_ethnic_prob() to get location_ethnic_prob
              and ethnic_perc
           4. Use preprocess_voter() to get cleaned voter's file
           5. Specify the column containing surname and geolocation code.
              eg, 'lastname' and 'gisjoin00'
              Please note that geolocation code in voter's file should match
              with census file.
           6. Put all of data described above into predict_ethnic() to get
              the predicted ethnic
           7. Use metric to measure the performance if needed.
    :return:
    """
    name_prob = preprocess_surname('./data/surname_list/app_c.csv')
    census = preprocess_census(
        './data/Census2000_BG/nhgis0061_ds147_2000_block.csv',
        transform=False, census_type='block')
    location_ethnic_prob, ethnic_perc = create_location_ethnic_prob(
        census, True)

    print('READ VOTER FILE')
    voter_file = preprocess_voter('./data/FL1_voters_geo_covariates.csv', census_type='block', sample=1000)
    print('READ OK')
    print('Sample size %d' % len(voter_file))
    surname = voter_file['lastname']
    cbg2000 = voter_file['gisjoin00']
    predict = predict_ethnic(
        surname, cbg2000, name_prob, location_ethnic_prob, False)
    predict = pd.Series(predict).apply(transform_output)
    predict.to_pickle('./out_blk.pkl')
    print(accuracy_score(predict, voter_file['race']))
    print(np.sum(predict == voter_file['race']) / float(len(predict)))
    print(classification_report(predict, voter_file['race']))
    print(confusion_matrix(predict, voter_file['race']))


if __name__ == '__main__':
    main()
