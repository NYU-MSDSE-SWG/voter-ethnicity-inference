from preprocessing import *
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def transform_output(x):
    if x == 'white':
        return 5
    elif x == 'black':
        return 3
    elif x == 'aian':
        return 1
    elif x == 'api':
        return 2
    elif x == 'hispanic':
        return 4
    elif x == '2race':
        return 7


def predict_ethnic(lastname, cbg2000, name_prob, location_prob, location_ethnic_prob, ethnic_perc, verbose_prob = False):
    lastname, cbg2000 = validate_input(lastname, cbg2000)
    name_p = name_prob.loc[lastname][['white','black','api','aian','2race','hispanic']]
    location_ethnic_p = location_ethnic_prob.loc[cbg2000][['white','black','api','aian','2race','hispanic']]
    location_p = location_prob.loc[cbg2000][['white','black','api','aian','2race','hispanic']]
    location_perc = location_prob.loc[cbg2000]['perc']
    ethnic_pred_prob = []
    ethnic_pred_race = []
    for i in range(len(lastname)):
        ans = (location_perc.iloc[i] * location_p.iloc[i] / ethnic_perc * name_p.iloc[i] /
                                 (name_p.iloc[i] * location_ethnic_p.iloc[i]).sum()).fillna(0)
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
    voter_file = pd.read_stata('./data/fl_voters_geo_covariates.dta', preserve_dtypes=False,
                               convert_categoricals=False, convert_dates=False)
    voter_file = preprocess_voter(voter_file)
    print('Sample size %d' %len(voter_file))
    surname = voter_file['lastname']
    cbg2000 = voter_file['bctcb2000']
    predict = predict_ethnic(surname, cbg2000, name_prob, location_prob, location_ethnic_prob, ethnic_perc, False)
    predict = pd.Series(predict).apply(transform_output)
    print(accuracy_score(predict, voter_file['race']))
    print(classification_report(predict, voter_file['race']))
    print(confusion_matrix(predict, voter_file['race']))