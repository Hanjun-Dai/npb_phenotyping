-------------------------------------------------------------------------------------------
Files:

mimic2-tensor-data.dat: sparse tensor for mimic2 dataset.
patDict.csv: patientID, patient coordinate in tensor
diagDict.csv: icdcode, diagnosis coordinate in tensor
medDict.csv: medication naem, medication coordinate in tensor 


-------------------------------------------------------------------------------------------
Note:

1. The created tensor is a sparse tensor, and its class is sptensor. 

2. To load the tensor, you can import sptensor and use sptensor.loadTensor(filename).

3. The tensor has three dimension. The first dimension denotes patients, the second dimension    denotes diagnosis, and the third one denotes medication.

