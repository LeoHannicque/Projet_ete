import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc

    

from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate0,sig0) = wav.read("0_george_0.wav")
mfcc_feat0 = mfcc(sig0,rate0)
mfcc_feat0 = preprocessing.scale(mfcc_feat0) #normalize the data between -1 and +1

(rate1,sig1) = wav.read("0_jackson_22.wav")
mfcc_feat1 = mfcc(sig1,rate1)
mfcc_feat1 = preprocessing.scale(mfcc_feat1) #normalize the data between -1 and +1


(rate2,sig2) = wav.read("9_yweweler_23.wav")
mfcc_feat2 = mfcc(sig2,rate2)
mfcc_feat2 = preprocessing.scale(mfcc_feat2) #normalize the data between -1 and +1



#delta = calculate_delta(mfcc_feat)
#combined = np.hstack((mfcc_feat,delta)) 


def dtw(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(s[i-1,:6]-t[j-1,:6],2)
            #cost = abs(s[i-1] - t[j-1])

            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    #print(dtw_matrix)
    return dtw_matrix

M01=dtw(mfcc_feat0,mfcc_feat1)
M02=dtw(mfcc_feat0,mfcc_feat2)
M12=dtw(mfcc_feat1,mfcc_feat2)
print(M01[len(mfcc_feat0),len(mfcc_feat1)]/(len(mfcc_feat0)+len(mfcc_feat1)))
print(M02[len(mfcc_feat0),len(mfcc_feat2)]/(len(mfcc_feat0)+len(mfcc_feat2)))
print(M12[len(mfcc_feat1),len(mfcc_feat2)]/(len(mfcc_feat1)+len(mfcc_feat2)))



