from preprocessing import *


def predict_ethnic(lastname, cbg2000, name_prob, location_prob, location_ethnic_prob, ethnic_perc):
    name_p = name_prob.loc[lastname][['white','black','api','aian','2race','hispanic']]
    location_ethnic_p = location_ethnic_prob.loc[cbg2000][['white','black','api','aian','2race','hispanic']]
    location_p = location_prob.loc[cbg2000][['white','black','api','aian','2race','hispanic']]
    location_perc = location_prob.loc[cbg2000]['perc']
    ethnic_prediction = location_perc * location_p / ethnic_perc * name_p /(name_p * location_ethnic_p).sum()
    return ethnic_prediction.fillna(0)


if __name__ == '__main__':
    name_prob = preprocess_surname('./data/surname_list/app_c.csv')
    census = preprocess_census('./data/Census2000_BG/C2000_NY.csv')
    location_prob = create_location_prob(census)
    location_ethnic_prob, ethnic_perc = create_location_ethnic_prob(location_prob, True)
    surname = 'WANG'
    cbg2000 = '20462011'

    predict = predict_ethnic(surname, cbg2000, name_prob, location_prob, location_ethnic_prob, ethnic_perc)
    print(predict)
    print(predict.argmax())
