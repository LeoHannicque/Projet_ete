import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc

    

from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("0_george_0.wav")
mfcc_feat = mfcc(sig,rate)
#mfcc_feat = preprocessing.scale(mfcc_feat)
#delta = calculate_delta(mfcc_feat)
#combined = np.hstack((mfcc_feat,delta)) 



