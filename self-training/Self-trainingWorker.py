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
        'worker': ['IP1:20200', 'IP2:20201']#workers are 2 servers with GPU.
    },
    'task': {'type': 'worker', 'index': 1}#1 because it is worker and different for each worker.
})

communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=communication_options)

print('Number of devices: %d' % strategy.num_replicas_in_sync) 
###################################

#directory to save files
mainDirect = '/home/edgar/Edgar/Semestre 4/legw-multi-worker-fewExamples/experiment16/part3/'

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
pathTrain = '/home/edgar/tensorflow_datasets/quickdraw_bitmap/edgarShuffled/train/'
pathTest = '/home/edgar/tensorflow_datasets/quickdraw_bitmap/edgarShuffled/test/'

#loading data for supervised learning
train = tf.data.experimental.load(pathTrain, element_spec=OrigTrain.element_spec)
test = tf.data.experimental.load(pathTest, element_spec=OrigTrain.element_spec)

#labeled data Pre-processing: normalization and shape setting.
train, test = massMethods.supervisedPreProcessing(train,test,shape)

#Dividing dataset for supervised training: taking the smaller part as labeled data.
supTrain = train.take(116690)

#Dividing dataset: taking the bigger part as unlabeled data to be pseudo-labeled.
sourceUnlabeled = train.skip(116690)
sourceUnlabeled = sourceUnlabeled.map(lambda image, label: (image),num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
unlabeled = sourceUnlabeled.take(50000)#Taking a portion of unlebeled data for efficient pseudo-labeling.

#experiments of diff amount of few labeled examples
setTrueLabels = np.array([2310, 3500, 4690, 5810, 7000, 8190, 9310, 10500, 11690, 23310, 35000, 46690, 58310, 70000, 81690, 93310, 105000, 116690]) #Quickdraw

#self-training method
massMethods.selfTraining(supTrain,
             test,
             unlabeled,
             sourceUnlabeled,
             setTrueLabels,
             mainDirect,
             strategy,
             numClasses = 10,
             Epochs = 50,
             verb = 1)
