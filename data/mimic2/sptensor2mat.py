import numpy as np
import scipy.io
import tools;

if __name__== "__main__":
    infile = file("result/mimic2-tensor-data.dat", "rb")
    subs = np.load(infile)
    vals = np.load(infile)
    
    matfile = "result/mimic2-tensor-data.mat"
    scipy.io.savemat(matfile, mdict={'A': subs[:,0], 'B': subs[:,1], 'C': subs[:,2]})
