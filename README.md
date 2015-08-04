# Voter’s Ethnicity Inference
------
This is a python program to predict voter’s ethnicity based on individual background information obtained from Voter Registration Records.

Voter Registration Record contains voter’s location, name, age and gender.
Example:

countyemsid | bctcb2000 | lastname | firstname | birthdate | gender
--- | --- | --- | --- | --- | ---
0000000Y | 50020011 | NASSAR | LINDA | 19481215 | F

where countyemsid is voter’s unique id, and bctcb2000 is geocoded location feature.

Combining datasets from Census, we implement method proposed by Kosuke Imai and Kabir Khanna(imai.princeton.edu/research/files/race.pdf).

- Census Frequently Occurring Surnames list can be obtained from http://www.census.gov/topics/population/genealogy/data/2000_surnames.html
- Census 2000 census block group level dataset can be obtained from http://old.socialexplorer.com/pub/reportdata/GeoSelection.aspx?reportid=R10950075
- Census 2000 census block level dataset can be obtained from NHGIS 
http://www.nhgis.org

# Result
Test on Florida's voter file, and drop voters whose surname are not in census surname list. 2010 Census dataset is used to provide geolocation information.

- surname-only prediction accuracy: 78.0%
- surname & census block prediction accuracy: 83.5%

## ROC Curve
![alt text](https://raw.githubusercontent.com/NYU-MSDSI/voter-ethnicity-inference/master/roc_curve/black.png)
![alt text](https://raw.githubusercontent.com/NYU-MSDSI/voter-ethnicity-inference/master/roc_curve/white.png)
![alt text](https://raw.githubusercontent.com/NYU-MSDSI/voter-ethnicity-inference/master/roc_curve/hispanic.png)



