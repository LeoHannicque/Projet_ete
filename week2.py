import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc

    

from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("0_george_0.wav")
mfcc_feat = mfcc(sig,rate)
mfcc_feat = preprocessing.scale(mfcc_feat) #normalize the data between -1 and +1

print(mfcc_feat)
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
            cost = abs(s[i-1] - t[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix



