import pandas as pd
import numpy as np
from sklearn.externals import joblib

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc

from preprocessing import (validate_input, preprocess_surname,
                           preprocess_census, preprocess_voter,
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
        ['white', 'black', 'asian', 'other', 'hispanic']]
    if len(cbg2000) != 0:
        location_ethnic_p = location_ethnic_prob.loc[cbg2000][
            ['white', 'black', 'asian', 'other', 'hispanic']]
        name_p = name_p.reset_index().drop('name', axis=1)
        location_ethnic_p = location_ethnic_p.reset_index().drop('GISJOIN', axis=1)
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
    Note: If cbg2000 is an empty list, it will only use name to do prediction
    """
    print('READ SURNAME LIST')
    name_prob = preprocess_surname('./data/surname_list/app_c.csv')

    print('READ CENSUS FILE')
    census = preprocess_census(
        './data/Census2000_BG/nhgis0062_ds172_2010_block.csv',
        transform=False, census_type='block')

    print('CREATE PROBABILITY MATRIX BASED ON CENSUS FILE')
    location_ethnic_prob, ethnic_perc = create_location_ethnic_prob(
        census, True)

    print('READ VOTER FILE')
    # If remove_flag == True, it will remove voters whose surname are not in census surname list
    # 
    # If remove_flag == False, it will keep those voters and use n-gram + logistic regression
    # to predict their ethnicity based on surname
    remove_flag = False
    voter_file = preprocess_voter('./data/FL1_voters_geo_covariates.csv', census_type='block', sample=0, remove_name=remove_flag)

    if not remove_flag:
        print('USE N-GRAM TO PREDICT VOTER NOT ON THE NAME LIST')
        notinlistname = np.setdiff1d(voter_file['lastname'], name_prob.index)
        try:
            n_gram_model = joblib.load('./model/n_gram.pkl')
            classifier = joblib.load('./model/classifier.pkl')
        except:
            n_gram_model, classifier = create_name_predictor('./data/surname_list/app_c.csv')
        notinname_prob = classifier.predict_proba(n_gram_model.transform(notinlistname))
        notinname_df = pd.DataFrame(notinname_prob, columns=classifier.classes_,index=notinlistname)
        notinname_df.index.name = 'name'
        name_prob = name_prob.append(notinname_df)
    print('Sample size %d' % len(voter_file))

    print('START PREDICTING')
    surname = voter_file['lastname']
    cbg2000 = voter_file['gisjoin10']
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
