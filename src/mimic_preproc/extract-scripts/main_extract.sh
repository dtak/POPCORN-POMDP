#!/bin/bash
#
# main_extract.sh
# Author: Joe Futoma
# Created: 5/28/2020
#
# Run this bash script from inside a directory containing flat csv's for 
# mimic-iii-v14 to extract a bunch of stuff from the 
# raw mimic data (assumed to be in a folder called 'raw_data')
#
# Files inside raw_data that make up MIMIC-III-V1.4:
#	ADMISSIONS.csv
#	CALLOUT.csv
#	CAREGIVERS.csv
#	CHARTEVENTS.csv
#	CPTEVENTS.csv
#	DATETIMEEVENTS.csv
#	D_CPT.csv
#	DIAGNOSES_ICD.csv
#	D_ICD_DIAGNOSES.csv
#	D_ICD_PROCEDURES.csv
#	D_ITEMS.csv
#	D_LABITEMS.csv
#	DRGCODES.csv
#	ICUSTAYS.csv
#	INPUTEVENTS_CV.csv
#	INPUTEVENTS_MV.csv
#	LABEVENTS.csv
#	MICROBIOLOGYEVENTS.csv
#	NOTEEVENTS.csv
#	OUTPUTEVENTS.csv
#	PATIENTS.csv
#	PRESCRIPTIONS.csv
#	PROCEDUREEVENTS_MV.csv
#	PROCEDURES_ICD.csv
#	SERVICES.csv
#	TRANSFERS.csv
#
# The main files (along with schema) of interest  I've used so far are:
#		ADMISSIONS: info on hospital admissions; one hospital admit can map to several ICU stays
#			"ROW_ID","SUBJECT_ID","HADM_ID","ADMITTIME","DISCHTIME","DEATHTIME","ADMISSION_TYPE",
#			"ADMISSION_LOCATION","DISCHARGE_LOCATION","INSURANCE","LANGUAGE","RELIGION",
#			"MARITAL_STATUS","ETHNICITY","EDREGTIME","EDOUTTIME","DIAGNOSIS",
#			"HOSPITAL_EXPIRE_FLAG","HAS_CHARTEVENTS_DATA"
#		CHARTEVENTS: gigantic table with all vitals, some labs, other stuff
#			"ROW_ID","SUBJECT_ID","HADM_ID","ICUSTAY_ID","ITEMID","CHARTTIME","STORETIME",
#			"CGID","VALUE","VALUENUM","VALUEUOM","WARNING","ERROR","RESULTSTATUS","STOPPED"
#		D_ITEMS: tells you what the stuff in CHARTEVENTS is if you need to find something
#			"ROW_ID","ITEMID","LABEL","ABBREVIATION","DBSOURCE","LINKSTO","CATEGORY",
#			"UNITNAME","PARAM_TYPE","CONCEPTID"
#		D_LABITEMS: similarly for labs in LABEVENTS
#			"ROW_ID","ITEMID","LABEL","FLUID","CATEGORY","LOINC_CODE"
#		ICUSTAYS: info on ICU stays, typically the unit of interest
#			"ROW_ID","SUBJECT_ID","HADM_ID","ICUSTAY_ID","DBSOURCE","FIRST_CAREUNIT",
#			"LAST_CAREUNIT","FIRST_WARDID","LAST_WARDID","INTIME","OUTTIME","LOS"
#		INPUTEVENTS_MV: meds & other stuff done, but *only* for MetaVision, ie newer data
#			"ROW_ID","SUBJECT_ID","HADM_ID","ICUSTAY_ID","STARTTIME","ENDTIME","ITEMID",
#			"AMOUNT","AMOUNTUOM","RATE","RATEUOM","STORETIME","CGID","ORDERID","LINKORDERID",
#			"ORDERCATEGORYNAME","SECONDARYORDERCATEGORYNAME","ORDERCOMPONENTTYPEDESCRIPTION",
#			"ORDERCATEGORYDESCRIPTION","PATIENTWEIGHT","TOTALAMOUNT","TOTALAMOUNTUOM","ISOPENBAG",
#			"CONTINUEINNEXTDEPT","CANCELREASON","STATUSDESCRIPTION","COMMENTS_EDITEDBY",
#			"COMMENTS_CANCELEDBY","COMMENTS_DATE","ORIGINALAMOUNT","ORIGINALRATE"
#		LABEVENTS: labs are mostly in here
#			"ROW_ID","SUBJECT_ID","HADM_ID","ITEMID","CHARTTIME","VALUE","VALUENUM","VALUEUOM","FLAG"
#		OUTPUTEVENTS: a few random useful things in here like urine output
#			"ROW_ID","SUBJECT_ID","HADM_ID","ICUSTAY_ID","CHARTTIME","ITEMID",
#			"VALUE","VALUEUOM","STORETIME","CGID","STOPPED","NEWBOTTLE","ISERROR"
#
# I've shifted to only using the more recent subset of the data, from MetaVision
# This was largely because I was pretty sketched out by the lack of "end_times"
# for treatments done in the INPUTEVENTS_CV table; unlike MV it just lists times treatments
# are started, but for doing RL you also really need to know when things are stopped.
# Maybe I'm missing something and they're buried in CV somewhere...
# Only using MV has the nice side effect that your scripts are half as long; CV has
# its own totally independent set of item id's to find all the stuff you generally want.
#
# TODO: should probably figure out how to get ventilation, that's important and 
#   I've been overlooking for ages... 
#   https://github.com/dtak/AMIA-2017-private/blob/master/mimic_iii_setup/customscripts/ventilation-durations-custom.sql
#
# NOTE: vasopressors and fluids can be pulled straight from python in a different script...

REPO_DATA_PATH='XXX'
cd $REPO_DATA_PATH
echo 'Moved to data folder in repo, '$REPO_DATA_PATH


#headers for stuff from specific files; will cut after awk'ing to get relevant columns
CHART_EVENTS_HEADER='"SUBJECT_ID","HADM_ID","ICUSTAY_ID","ITEMID","CHARTTIME","VALUE","VALUENUM","VALUEUOM"'
LAB_EVENTS_HEADER='"SUBJECT_ID","HADM_ID","ITEMID","CHARTTIME","VALUE","VALUENUM","VALUEUOM"'
PROCEDURE_EVENTS_HEADER='"SUBJECT_ID","HADM_ID","ICUSTAY_ID","STARTTIME","ENDTIME","ITEMID"'
MICROBIO_EVENTS_HEADER='SUBJECT_ID","HADM_ID","CHARTDATE","CHARTTIME","SPEC_ITEMID","SPEC_TYPE_DESC"'
OUTPUT_EVENTS_HEADER='"SUBJECT_ID","HADM_ID","ICUSTAY_ID","CHARTTIME","ITEMID","VALUE","VALUEUOM"'

###
### get vitals & labs from chartevents...
###

echo 'Getting stuff from CHARTEVENTS...'

echo 'Getting SBP...'
echo $CHART_EVENTS_HEADER > query-data/sbp.csv
awk -F',' '($5==224167 || $5==227243 || $5==220179 || $5==220050 || $5==225309)' raw-data/CHARTEVENTS.csv | cut -d, -f2,3,4,5,6,9,10,11 >> query-data/sbp.csv

echo 'Getting DBP...'
echo $CHART_EVENTS_HEADER > query-data/dbp.csv
awk -F',' '($5==227242 || $5==224643 || $5==220180 || $5==220051 || $5==225310)' raw-data/CHARTEVENTS.csv | cut -d, -f2,3,4,5,6,9,10,11 >> query-data/dbp.csv

echo 'Getting MAP...'
echo $CHART_EVENTS_HEADER > query-data/map.csv
awk -F',' '($5==220181 || $5==220052 || $5==225312)' raw-data/CHARTEVENTS.csv | cut -d, -f2,3,4,5,6,9,10,11 >> query-data/map.csv

echo 'Getting Weight...'
echo $CHART_EVENTS_HEADER > query-data/weight.csv
awk -F',' '($5==224639 || $5==226512)' raw-data/CHARTEVENTS.csv | cut -d, -f2,3,4,5,6,9,10,11 >> query-data/weight.csv

echo 'Getting GCS...'
echo $CHART_EVENTS_HEADER > query-data/GCS.csv
awk -F',' '($5==227011 || $5==227012 || $5==227014 || $5==228112 || $5==226756 || $5==226757 || $5==226758 || $5==220739 || $5==223900 || $5==223901)' raw-data/CHARTEVENTS.csv | cut -d, -f2,3,4,5,6,9,10,11 >> query-data/GCS.csv

echo 'Getting Total Bilirubin (CE)...'
echo $CHART_EVENTS_HEADER > query-data/bilirubin_total_ce.csv
awk -F',' '($5==225690)' raw-data/CHARTEVENTS.csv | cut -d, -f2,3,4,5,6,9,10,11 >> query-data/bilirubin_total_ce.csv

echo 'Getting INR (CE)...'
echo $CHART_EVENTS_HEADER > query-data/inr_ce.csv
awk -F',' '($5==227467)' raw-data/CHARTEVENTS.csv | cut -d, -f2,3,4,5,6,9,10,11 >> query-data/inr_ce.csv

echo 'Getting Hgb (CE)...'
echo $CHART_EVENTS_HEADER > query-data/hgb_ce.csv
awk -F',' '($5==220228)' raw-data/CHARTEVENTS.csv | cut -d, -f2,3,4,5,6,9,10,11 >> query-data/hgb_ce.csv

echo 'Getting HR...'
echo $CHART_EVENTS_HEADER > query-data/hr.csv
awk -F',' '($5==211 || $5==220045)' raw-data/CHARTEVENTS.csv | cut -d, -f2,3,4,5,6,9,10,11 >> query-data/hr.csv

echo 'Getting Glucose (CE)...'
echo $CHART_EVENTS_HEADER > query-data/glucose_ce.csv
awk -F',' '($5==807 || $5==811 || $5==1529 || $5==3745 || $5==3744 || $5==225664 || $5==220621 || $5==226537)' raw-data/CHARTEVENTS.csv | cut -d, -f2,3,4,5,6,9,10,11 >> query-data/glucose_ce.csv

echo 'Getting FiO2...'
echo $CHART_EVENTS_HEADER > query-data/fio2.csv
awk -F',' '($5==190 || $5==3420 || $5==223835)' raw-data/CHARTEVENTS.csv | cut -d, -f2,3,4,5,6,9,10,11 >> query-data/fio2.csv

echo 'Getting SpO2...'
echo $CHART_EVENTS_HEADER > query-data/spo2.csv
awk -F',' '($5==646 || $5==220277)' raw-data/CHARTEVENTS.csv | cut -d, -f2,3,4,5,6,9,10,11 >> query-data/spo2.csv

echo 'Getting RR...'
echo $CHART_EVENTS_HEADER > query-data/spontaneousrr.csv
awk -F',' '($5==615 || $5==618 || $5==220210 || $5==224690)' raw-data/CHARTEVENTS.csv | cut -d, -f2,3,4,5,6,9,10,11 >> query-data/spontaneousrr.csv

echo 'Getting Temp...'
echo $CHART_EVENTS_HEADER > query-data/temp.csv
#NOTE: first 2 ids are C, second 2 are F; C = (F-32)/1.8
awk -F',' '($5==223762 || $5==676 || $5==223761 || $5==678)' raw-data/CHARTEVENTS.csv | cut -d, -f2,3,4,5,6,9,10,11 >> query-data/temp.csv

###
### get labs from labevents
###

echo 'Getting stuff from LABEVENTS...'

echo 'Getting ALT...'
echo $LAB_EVENTS_HEADER > query-data/alt.csv
awk -F',' '($4==50861)' raw-data/LABEVENTS.csv | cut -d, -f2-8 >> query-data/alt.csv

echo 'Getting AST...'
echo $LAB_EVENTS_HEADER > query-data/ast.csv
awk -F',' '($4==50878)' raw-data/LABEVENTS.csv | cut -d, -f2-8 >> query-data/ast.csv

echo 'Getting Total Bilirubin (LE)...'
echo $LAB_EVENTS_HEADER > query-data/bilirubin_total_labs.csv
awk -F',' '($4==50885)' raw-data/LABEVENTS.csv | cut -d, -f2-8 >> query-data/bilirubin_total_labs.csv

### NOTE: this var overlaps with the exact same item id as in bicarb...should probably only keep one...
echo 'CO2...'
echo $LAB_EVENTS_HEADER > query-data/co2.csv
awk -F',' '($4==50804)' raw-data/LABEVENTS.csv | cut -d, -f2-8 >> query-data/co2.csv

echo 'Hgb (LE)...'
echo $LAB_EVENTS_HEADER > query-data/hgb_labs.csv
awk -F',' '($4==51222 || $4==50811)' raw-data/LABEVENTS.csv | cut -d, -f2-8 >> query-data/hgb_labs.csv

echo 'PCO2...'
echo $LAB_EVENTS_HEADER > query-data/pco2.csv
awk -F',' '($4==50818)' raw-data/LABEVENTS.csv | cut -d, -f2-8 >> query-data/pco2.csv

echo 'PO2...'
echo $LAB_EVENTS_HEADER > query-data/po2.csv
awk -F',' '($4==50821)' raw-data/LABEVENTS.csv | cut -d, -f2-8 >> query-data/po2.csv

echo 'INR (LE)...'
echo $LAB_EVENTS_HEADER > query-data/inr_labs.csv
awk -F',' '($4==51237)' raw-data/LABEVENTS.csv | cut -d, -f2-8 >> query-data/inr_labs.csv

echo 'WBC...'
echo $LAB_EVENTS_HEADER > query-data/wbc.csv
awk -F',' '($4==51300 || $4==51301)' raw-data/LABEVENTS.csv | cut -d, -f2-8 >> query-data/wbc.csv

echo 'Lactate...'
echo $LAB_EVENTS_HEADER > query-data/lactate.csv
awk -F',' '($4==50813)' raw-data/LABEVENTS.csv | cut -d, -f2-8 >> query-data/lactate.csv

echo 'Magnesium...'
echo $LAB_EVENTS_HEADER > query-data/magnesium.csv
awk -F',' '($4==8321 || $4==1970 || $4==6133 || $4==1532 || $4==220635 || $4==50960)' raw-data/LABEVENTS.csv | cut -d, -f2-8 >> query-data/magnesium.csv

echo 'Platelets...'
echo $LAB_EVENTS_HEADER > query-data/platelets.csv
awk -F',' '($4==51265)' raw-data/LABEVENTS.csv | cut -d, -f2-8 >> query-data/platelets.csv

echo 'Bicarbonate...'
echo $LAB_EVENTS_HEADER > query-data/bicarbonate.csv
awk -F',' '($4==50882 || $4==50804 || $4==50803)' raw-data/LABEVENTS.csv | cut -d, -f2-8 >> query-data/bicarbonate.csv

echo 'Hct...'
echo $LAB_EVENTS_HEADER > query-data/hct.csv
awk -F',' '($4==50810 || $4==51221)' raw-data/LABEVENTS.csv | cut -d, -f2-8 >> query-data/hct.csv

echo 'Glucose (LE)...'
echo $LAB_EVENTS_HEADER > query-data/glucose_labs.csv
awk -F',' '($4==50809 || $4==50931)' raw-data/LABEVENTS.csv | cut -d, -f2-8 >> query-data/glucose_labs.csv

echo 'Creatinine...'
echo $LAB_EVENTS_HEADER > query-data/creatinine.csv
awk -F',' '($4==50912)' raw-data/LABEVENTS.csv | cut -d, -f2-8 >> query-data/creatinine.csv

echo 'BUN...'
echo $LAB_EVENTS_HEADER > query-data/bun.csv
awk -F',' '($4==51006)' raw-data/LABEVENTS.csv | cut -d, -f2-8 >> query-data/bun.csv

echo 'Potassium...'
echo $LAB_EVENTS_HEADER > query-data/potassium.csv
awk -F',' '($4==50971 || $4==50822)' raw-data/LABEVENTS.csv | cut -d, -f2-8 >> query-data/potassium.csv

echo 'Sodium...'
echo $LAB_EVENTS_HEADER > query-data/sodium.csv
awk -F',' '($4==50824 || $4==50983)' raw-data/LABEVENTS.csv | cut -d, -f2-8 >> query-data/sodium.csv

###
### get urine out from output events
###

echo 'Getting Urine Output from Output Events...'
echo $OUTPUT_EVENTS_HEADER > query-data/urine.csv
awk -F',' '($6==40055 || $6==43175 || $6==40069 || $6==40094 || $6==40715 || $6==40473 || $6==40085 || $6==40057 || $6==40056 || $6==40405 || $6==40428 || $6==40086 || $6==40096 || $6==40651 || $6==226559 || $6==226560 || $6==227510 || $6==226561 || $6==226584 || $6==226563 || $6==226564 || $6==226565 || $6==226567  || $6==226557  || $6==226558)' raw-data/OUTPUTEVENTS.csv | cut -d, -f2-8 >> query-data/urine.csv

###
### times of blood cultures can be useful as a proxy for suspicion of infection...
###

echo 'Getting Blood Cultures (PE)...'
echo $PROCEDURE_EVENTS_HEADER > query-data/bloodculture_pe.csv
awk -F',' '($7==225401 || $7==225444)' raw-data/PROCEDUREEVENTS_MV.csv | cut -d, -f2-7 >> query-data/bloodculture_pe.csv

echo 'Getting Blood Cultures (Micro)...'
echo $MICROBIO_EVENTS_HEADER > query-data/bloodculture_micro.csv
awk -F',' '($6==70011 || $6==70012)' raw-data/MICROBIOLOGYEVENTS.csv | cut -d, -f2-7 >> query-data/bloodculture_micro.csv 

### we will handle population & baseline admission data in data cleaning python script
### for now, just grab all MV icuid's for easy access later

#if they had *no* input events they're probably too short/simple an ICU stay to be useful...
echo 'ICUSTAY_ID' > query-data/mv_icuids.csv
cut -d, -f4 raw-data/INPUTEVENTS_MV.csv | sort | uniq >> query-data/mv_icuids.csv
sed -i '$d' query-data/mv_icuids.csv #uniq spits out "ICUSTAY_ID" as last line, so del
sed -i '2d' query-data/mv_icuids.csv #second line of file is blank space from uniq, so del






