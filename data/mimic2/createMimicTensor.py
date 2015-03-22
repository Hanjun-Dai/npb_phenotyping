import pandas as pd 
import numpy as np 
from sklearn.utils import extmath
import sptensor
import csv
from collections import OrderedDict

def createIndexMap(key_array):
    keys = np.unique(key_array)
    key_index_list = zip(keys, np.arange(keys.size))
    key_index_map = dict((key,value) for (key,value) in key_index_list)
    key_index_ordered_map = OrderedDict(sorted(key_index_map.items(), key=lambda x: x[1]))
    return key_index_ordered_map

def find_nearest_diag_date(input_date, input_subject_id, diag_row_index):
    diag_date_array = diag_row_index[1, diag_row_index[0,:]==input_subject_id]
    if diag_date_array.size==0:
        return None
    date_distance = input_date - diag_date_array
    date_distance = np.where(date_distance>=0, date_distance, 100)
    if np.min(date_distance) < 7:
        return diag_date_array[np.argmin(date_distance)]
    else:
        return None

def compute_relative_date(date_array, start = '2500-06-22'):
    start_date = np.datetime64(start)
    new_date_array = date_array.apply(lambda x: long((x.astype('M8[D]')-start_date)/np.timedelta64(1, 'D')))
    return new_date_array

def diag_cross_med(med_file, diag_file):
    med = pd.read_csv(med_file, 
		usecols=['subject_id','med_name','date'], 
		parse_dates=["date"], 
		date_parser=np.datetime64)
    diag = pd.read_csv(diag_file, 
		usecols=['subject_id', 'code', 'date'], 
		parse_dates=['date'], 
		date_parser=np.datetime64) 
  
	## Compute the relative date
    med.date= compute_relative_date(med.date)
    diag.date = compute_relative_date(diag.date)
    
    diag.to_csv("result/diag.csv", index=False)    
    
    print('Finished computing relative date')
    ## group diagnosis records by subject id and date
    diag = diag.groupby(['subject_id','date']).apply(lambda x: np.unique(x.code))
	## group medication records by subject id and assigned date
    diag_row_index = np.array(zip(*diag.index.values))
    med['assignment'] = med.apply(lambda x: find_nearest_diag_date(x.date, x.subject_id, diag_row_index), axis=1)
    
    med.to_csv("result/med.csv", index=False)    
    
    med = med.dropna()
    med = med.loc[:,['subject_id', 'assignment', 'med_name']]
    med.columns = ['subject_id', 'date', 'med_name']
    med = med.groupby(['subject_id', 'date']).apply(lambda x: np.unique(x.med_name))
    
    print('Preparation done.')
    diag_med_comb = pd.concat([diag, med], axis=1)
    diag_med_comb = diag_med_comb.dropna()
    diag_med_comb.columns = ['code', 'med_name']            
    diag_med_comb['subject_id'] = zip(*(list(diag_med_comb.index)))[0]
    print('diag_med_comb done') 
    
    return diag_med_comb

def constructTensor(med_file, diag_file):    
    diag_med_comb = diag_cross_med(med_file, diag_file)
	## create index map for subject_id, icdcode, and med_name
    patDict = createIndexMap(diag_med_comb.subject_id)
    medDict = createIndexMap(np.hstack(diag_med_comb.med_name))
    diagDict = createIndexMap(np.hstack(diag_med_comb.code))
    
    tensorIdx = np.array([[0,0,0]])
    tensorVal = np.array([[0]])
    for i in xrange(diag_med_comb.shape[0]):
        curDiag = [diagDict[x] for x in diag_med_comb.iloc[i,0]]
        curMed = [medDict[x] for x in diag_med_comb.iloc[i,1]]
        curPatId = patDict[diag_med_comb.iloc[i,2]]
        dmCombo = extmath.cartesian((curDiag, curMed))
        tensorIdx = np.append(tensorIdx,np.column_stack((np.repeat(curPatId, dmCombo.shape[0]), dmCombo)),axis=0)
        tensorVal = np.append(tensorVal, np.ones((dmCombo.shape[0],1), dtype=np.int), axis=0)

    tensorIdx = np.delete(tensorIdx, (0), axis=0)
    tensorVal = np.delete(tensorVal, (0), axis=0)
    tenX = sptensor.sptensor(tensorIdx, tensorVal, np.array([len(patDict), len(diagDict), len(medDict)]))
    axisDict = {0: patDict, 1: diagDict, 2: medDict}
    
    return tenX, axisDict

def saveInf(dict, outfile):
    w = csv.writer(open(outfile, "w"))
    for key, val in dict.items():
        w.writerow([key, val])
    return None
    
if __name__== "__main__":
    
    tenX, axisDict = constructTensor("data/mimic2_datadump_20150227/MEDICATION_JOINED_BOOLEAN.csv",
                    "data/mimic2_datadump_20150227/DIAGNOSTIC.csv")
    
    tenX.saveTensor("result/mimic2-tensor-data.dat")
    saveInf(axisDict[0], "result/patDict.csv")
    saveInf(axisDict[1], "result/diagDict.csv")
    saveInf(axisDict[2], "result/medDict.csv")


    
    



