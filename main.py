import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc

from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

from dbscan import DBSCAN

import os
from collections import defaultdict
from dbscan import DBSCAN


if __name__ == '__main__':
    
		
	for path, dirs,files in os.walk("recordings"):
		M=np.zeros(len(files))
		datasets = defaultdict(list)

		cnt=0
		for filename in files: #extract the mfcc coefficients from the audio files
			(rate,sig) = wav.read("recordings"+"/"+filename) 
			mfcc_feat = mfcc(sig,rate)
			mfcc_feat = preprocessing.scale(mfcc_feat) #normalize the data between -1 and +1
			datasets[cnt]=mfcc_feat
			cnt+=1


	min_points = 5 #parameters for DBSCAN
	eps = 0.3
	
	
	dbscan = DBSCAN(min_points, 0.3)  

	labels = dbscan.fit(datasets)
	print(labels)


  
