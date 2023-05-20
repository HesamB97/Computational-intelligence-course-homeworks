import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc
import _pickle as cPickle
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM 
import warnings
import webbrowser
import time
import os
warnings.filterwarnings("ignore")

#############################
#############################

# First Section Functions for Feature Extraction

def calculate_delta(array):
    """Calculate and returns the delta of given feature vector matrix"""

    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
              first =0
            else:
              first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def extract_features(audio,rate):
    """extract 20 dim mfcc features from an audio, performs CMS and combines 
    delta to make it 40 dim feature vector"""    
    
    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined

#############################
#############################

# Second Section: Funcotion and Code for Model Training


# path to training data
source   = "development_set/"
# source   = "trainingData/"   

#path where training speakers will be saved

# dest = "speaker_models/"
train_file = "development_set_enroll.txt"

dest = "Speakers_models/"
# train_file = "trainingDataPath.txt"        
file_paths = open(train_file,'r')

count = 1
# Extracting features for each speaker (5 files per speakers)
features = np.asarray(())
for path in file_paths:    
    path = path.strip()   
    print(path)
    
    # read the audio
    sr,audio = read(source + path)
    
    
    # extract 40 dimensional MFCC & delta MFCC features
    vector   = extract_features(audio,sr)
    
    
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # when features of 5 files of speaker are concatenated, then do model training
	# -> if count == 5: --> edited below
    if count == 15:    
        gmm = GMM(n_components = 16, covariance_type='diag',n_init = 3)
        gmm.fit(features)
        
        # dumping the trained gaussian model
        picklefile = path.split("-")[0]+".gmm"
        cPickle.dump(gmm,open(dest + picklefile,'wb'))
        print('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)    
        features = np.asarray(())
        count = 0
    count = count + 1

###############################
###############################

# Third Section: model performance analysis

#path to training data
source   = "development_set//"
modelpath = "Speakers_models"
test_file = "development_set_test.txt"
file_paths = open(test_file,'r')
 
gmm_files = [os.path.join(modelpath,fname) for fname in
              os.listdir(modelpath) if fname.endswith('.gmm')]
 
#Load the Gaussian gender Models
models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname
              in gmm_files]
 
# Read the test directory and get the list of test audio files
for path in file_paths:   
 
    path = path.strip()
    print(path)
    sr,audio = read(source + str(path))
    vector   = extract_features(audio,sr)
 
    log_likelihood = np.zeros(len(models)) 
 
    for i in range(len(models)):
        gmm    = models[i]  #checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
 
    winner = np.argmax(log_likelihood)
    print("\tdetected as - ", speakers[winner])
    # time.sleep(1.0)
