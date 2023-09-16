import os
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

from time import time
import tensorflow_datasets as tfds
from mypackages.learningSort import massiveDataMethods as massMethods

#Multiworker setting
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['132.247.11.16:20002', '132.247.11.91:20003']#workers are 2 servers with GPU.
    },
    'task': {'type': 'worker', 'index': 0}#0 because it is the chief and different for each worker.
})

communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
strategy_ = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=communication_options)

print('Number of devices: %d' % strategy_.num_replicas_in_sync)

###################################

#directory to save files
mainDirect_ = '/home/est1/Edgar/Semestre 4/model implementation/legw-multi-worker-fewExamples/semi-supervised layer-wise/quickdraw/experiment3/'
namefile_ = 'semi-supervisedQuickdraw' #file to write data
pretrainfile_ = 'pretrainingLayerWiseQuickdraw' #file to load weights
epochs_ = 1000#Early stopping is used really.
batch_size_ = 2048
lr_ = 0.001
warmupEpochs_ = 16
toAndFro_ = 583310#amount to pre-train with, quickdraw dataset.

#Dataset definition
OrigTrain = tfds.load(
    'quickdraw_bitmap',
    split='train',
    #batch_size=10000,# -1
    as_supervised=True,
    shuffle_files = True
)

#dataset specifications
numClasses_ = 10
shape = (28, 28)
pathTrain = '/home/est1/tensorflow_datasets/quickdraw_bitmap/edgarShuffled/train/'
pathTest = '/home/est1/tensorflow_datasets/quickdraw_bitmap/edgarShuffled/test/'

#UNSUPERVISED LEARNING

#loading data for Unsupervised learning
train = tf.data.experimental.load(pathTrain, element_spec=OrigTrain.element_spec)
test = tf.data.experimental.load(pathTest, element_spec=OrigTrain.element_spec)

#Unlabeled data Pre-processing: normalization and shape setting.
unsupTrain, unsupTest = massMethods.unsupervisedPreProcessing(train,test,shape)

#Dividing dataset in pre-training and training: taking the bigger part as unlabeled data.
unsupTrain = unsupTrain.take(toAndFro_)

#unsupervised learning: layer-wise pre-training.
massMethods.preTraining(unsupTrain,
               unsupTest,
               mainDirect = mainDirect_,
               namefile = pretrainfile_,
               strategy = strategy_,
               Epochs = epochs_,
               batchSize = batch_size_,
               lr = lr_,
               warmupEpoch = warmupEpochs_)#the function will compute the min warmup epochs based on 2048 bs and 16 warmup epochs.

#SUPERVISED LEARNING

#loading data for supervised
train = tf.data.experimental.load(pathTrain, element_spec=OrigTrain.element_spec)
test = tf.data.experimental.load(pathTest, element_spec=OrigTrain.element_spec)

#labeled data Pre-processing: normalization and shape setting.
supTrain, supTest = massMethods.supervisedPreProcessing(train,test,shape)

#Dividing dataset for supervised training: taking the smaller part as labeled data.
supTrain = supTrain.skip(toAndFro_)

#experiments of diff amount of few labeled examples
setTrueLabels_ = np.array([2310, 3500, 4690, 5810, 7000, 8190, 9310, 10500, 11690, 23310, 35000, 46690, 58310, 70000, 81690, 93310, 105000, 116690]) #Quickdraw

#supervised learning
massMethods.semiSupervised(supTrain,
                            supTest,
                            mainDirect = mainDirect_,
                            namefile = namefile_,
                            pretrainfile = pretrainfile_,
                            strategy = strategy_,
                            Epochs = 50,#50,epochs_,
                            numClasses = numClasses_,
                            batchSize = 32,
                            lr = 0.001,
                            warmupEpoch = 8,
                            verb = 1,
                            setTrueLabels = setTrueLabels_)
