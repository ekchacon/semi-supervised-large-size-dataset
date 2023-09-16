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
        'worker': ['132.247.11.16:20000', '132.247.11.91:20001']#workers are 2 servers with GPU.
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
mainDirect_ = '/home/est1/Edgar/Semestre 4/model implementation/legw-multi-worker-fewExamples/supervised/quickdraw/experiment2/'
namefile_ = 'supervisedQuickdraw' #file to write data
toAndFro_ = 583310#amount of unlabeled data for pre-training.

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
pathTrain = '/home/est1/tensorflow_datasets/quickdraw_bitmap/edgar/train/'
pathTest = '/home/est1/tensorflow_datasets/quickdraw_bitmap/edgar/test/'

#loading data for supervised learning
train = tf.data.experimental.load(pathTrain, element_spec=OrigTrain.element_spec)
test = tf.data.experimental.load(pathTest, element_spec=OrigTrain.element_spec)

#normalization and shape setting
train, test = massMethods.supervisedPreProcessing(train,test,shape)

#Shuffle
supTrain = train.shuffle(train.cardinality().numpy(), reshuffle_each_iteration=False)
supTest = test.shuffle(test.cardinality().numpy(), reshuffle_each_iteration=False)

#Dividing dataset and taking few labeled examples for training.
supTrain = supTrain.skip(toAndFro_)

#experiments of diff amount of few labeled examples
setTrueLabels_ = np.array([2310, 3500, 4690, 5810, 7000, 8190, 9310, 10500, 11690, 23310, 35000, 46690, 58310, 70000, 81690, 93310, 105000, 116690]) #Quickdraw
#setTrueLabels_ = np.array([81690, 93310, 105000, 116690]) #Quickdraw
#setTrueLabels_ = np.array([5810]) #Quickdraw

#supervised learning
massMethods.supervised(supTrain,
                            supTest,
                            mainDirect = mainDirect_,
                            namefile = namefile_,
                            strategy = strategy_,
                            Epochs = 50,
                            numClasses = numClasses_,
                            batchSize = 32,#not really used
                            lr = 0.001,#not really used
                            warmupEpoch = 8,#not really used
                            verb = 1,
                            setTrueLabels = setTrueLabels_)
