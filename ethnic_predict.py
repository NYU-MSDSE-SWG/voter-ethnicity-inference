from preprocessing import *
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import random

def transform_output(x):
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


def predict_ethnic(lastname, cbg2000, name_prob, location_ethnic_prob, verbose_prob = False):
    lastname, cbg2000 = validate_input(lastname, cbg2000)
    name_p = name_prob.loc[lastname][['white','black','asian','other','hispanic']]
    location_ethnic_p = location_ethnic_prob.loc[cbg2000][['white','black','asian','other','hispanic']]
    ethnic_pred_prob = []
    ethnic_pred_race = []
    for i in range(len(lastname)):
        numerator = location_ethnic_p.iloc[i] * name_p.iloc[i]
        denominator = (location_ethnic_p.iloc[i] * name_p.iloc[i]).sum()
        ans = (numerator / denominator).fillna(0)
        ethnic_pred_prob.append(ans)
        ethnic_pred_race.append(ans.argmax())
    if verbose_prob:
        return ethnic_pred_race, ethnic_pred_prob
    else:
        return ethnic_pred_race


if __name__ == '__main__':
    name_prob = preprocess_surname('./data/surname_list/app_c.csv')
    census = preprocess_census('./data/Census2000_BG/C2000_FL.csv', transform=False)
    location_prob = create_location_prob(census)
    location_ethnic_prob, ethnic_perc = create_location_ethnic_prob(location_prob, True)
    if True:
        print('READ VOTER FILE')
        voter_file = pd.read_csv('./data/test_input.csv', dtype=object)
        rows = random.sample(voter_file.index, 1000)
        voter_file = voter_file.ix[rows]
        voter_file['race'] = (voter_file['race'].astype(float)).astype(int)
        voter_file.loc[voter_file['race'] == 7, 'race'] = 6
        voter_file.loc[voter_file['race'] == 1, 'race'] = 6
        voter_file.loc[voter_file['race'] == 9, 'race'] = 6
        print('READ OK')
    else:
        print('READ VOTER FILE')
        voter_file = pd.read_stata('./data/fl_voters_geo_covariates.dta', preserve_dtypes=False,
                                   convert_categoricals=False, convert_dates=False)
        print('READ OK')
        voter_file = voter_file.iloc[:10000]
        voter_file = preprocess_voter(voter_file)

    print('Sample size %d' %len(voter_file))
    surname = voter_file['lastname']
    cbg2000 = voter_file['bctcb2000']
    predict = predict_ethnic(surname, cbg2000, name_prob, location_ethnic_prob, False)
    predict = pd.Series(predict).apply(transform_output)
    predict.to_pickle('./out.pkl')
    print(accuracy_score(predict, voter_file['race']))
    print(np.sum(predict == voter_file['race'])/float(len(predict)))
    print(classification_report(predict, voter_file['race']))
    print(confusion_matrix(predict, voter_file['race']))