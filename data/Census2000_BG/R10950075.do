/* 
* TODO: 1. Place the .txt data file and the dictionary file you downloaded in the work folder, or enter the full path to these files!
*       2. You may have to increase memory using the 'set mem' statement. It is commented out in the code bellow.
*
* If you have any questions or need assistance contact info@socialexplorer.com.
*/

///set mem 512m
set more off

global data "/Users/Kang/Google Drive/Data_US_NY_Census"

infile using "$data/R10950075.dct", using("$data/R10950075_SL150.txt") clear 

keep T001_001 T014_003 TRACT BLKGRP COUNTY

gen borocode = ""
replace borocode = "3" if COUNTY == "047"
replace borocode = "4" if COUNTY == "081"
replace borocode = "1" if COUNTY == "061"
replace borocode = "2" if COUNTY == "005"
replace borocode = "5" if COUNTY == "085"

keep if borocode != ""


gen cbg2000 = borocode + TRACT + BLKGRP
keep cbg2000 T001_001 T014_003
rename T001_001 total_pop
rename T014_003 black_pop 

save "$data/census2000_cbg", replace 