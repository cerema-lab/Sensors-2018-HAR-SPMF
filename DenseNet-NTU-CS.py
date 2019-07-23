# Huy-Hieu PHAM, Ph.D student
# Cerema & Institut de Recherche en Informatique de Toulouse (IRIT)
# Description: Training Inception V-4 model from human action recognition.
# Date: 24 / 11 / 2017
# Python 3.2.0, Keras 2.0 with Tensorflow backend.



# Import libraries and packages.

from __future__ import print_function

import os
import math

import numpy as np
np.random.seed()
import tensorflow as tf
from keras import optimizers
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, merge
from keras.engine import Input, Model
from keras.layers.advanced_activations import ELU
from keras.optimizers import SGD
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.utils import plot_model
import keras.backend as K
import json
import time
import argparse
import itertools
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix

# Learning rate schedule.
def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.5
	epochs_drop = 20
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


# Cross-Subject: 39788,15275 -- Cross-View 37273,18798
img_width, img_height = 32, 32
train_data_dir = 'data/NTU-RGB+D/Cross-Subject/train'
validation_data_dir = 'data/NTU-RGB+D/Cross-Subject/validation'
nb_train_samples = 39821
nb_validation_samples = 15275
epochs = 300
batch_size = 256

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# we reduce # filters by factor of 8 compared to original inception-v4
nb_filters_reduction_factor = 3

def inception_resnet_v2_stem(x):
    # in original inception-resnet-v2, conv stride is 2
    x = Convolution2D(32//nb_filters_reduction_factor, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='valid', dim_ordering='tf')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(32//nb_filters_reduction_factor, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='valid', dim_ordering='tf')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)				  
    x = Convolution2D(64//nb_filters_reduction_factor, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    # in original inception-resnet-v2, stride is 2
    a = MaxPooling2D((3, 3), strides=(1, 1), border_mode='valid', dim_ordering='tf')(x)
    # in original inception-resnet-v2, conv stride is 2
    b = Convolution2D(96//nb_filters_reduction_factor, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='valid', dim_ordering='tf')(x)
    b = BatchNormalization()(b)
    b = ELU()(b)				  
    x = merge([a, b], mode='concat', concat_axis=-1)
    
    a = Convolution2D(64//nb_filters_reduction_factor, 1, 1, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    a = BatchNormalization()(a)
    a = ELU()(a)
    a = Dropout(0.2)(a)
    a = Convolution2D(96//nb_filters_reduction_factor, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='valid', dim_ordering='tf')(a)
    a = BatchNormalization()(a)
    a = ELU()(a)				  
    b = Convolution2D(64//nb_filters_reduction_factor, 1, 1, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    b = BatchNormalization()(b)
    b = ELU()(b)				  
    b = Convolution2D(64//nb_filters_reduction_factor, 7, 1, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(b)
    b = BatchNormalization()(b)
    b = ELU()(b)
    b = Dropout(0.2)(b)
    b = Convolution2D(64//nb_filters_reduction_factor, 1, 7, subsample=(1, 1), 
                      init='he_normal', border_mode='same', dim_ordering='tf')(b)
    b = BatchNormalization()(b)
    b = ELU()(b)				  
    b = Convolution2D(96//nb_filters_reduction_factor, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='valid', dim_ordering='tf')(b)
    b = BatchNormalization()(b)
    b = ELU()(b)				  
    x = merge([a, b], mode='concat', concat_axis=-1)
    
    # in original inception-resnet-v2, conv stride should be 2
    a = Convolution2D(192//nb_filters_reduction_factor, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='valid', dim_ordering='tf')(x)
    a = BatchNormalization()(a)
    a = ELU()(a)				  
    # in original inception-resnet-v2, stride is 2
    b = MaxPooling2D((3, 3), strides=(1, 1), border_mode='valid', dim_ordering='tf')(x)
    x = merge([a, b], mode='concat', concat_axis=-1)
    
    x = ELU()(x)
    
    return x


def inception_resnet_v2_A(x):
    shortcut = x
    
    a = Convolution2D(32//nb_filters_reduction_factor, 1, 1, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    a = BatchNormalization()(a)
    a = ELU()(a)
    b = Convolution2D(32//nb_filters_reduction_factor, 1, 1, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    b = BatchNormalization()(b)
    b = ELU()(b)
    b = Dropout(0.2)(b)
    b = Convolution2D(32//nb_filters_reduction_factor, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(b)
    b = BatchNormalization()(b)
    b = ELU()(b)				  
    c = Convolution2D(32//nb_filters_reduction_factor, 1, 1, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    c = BatchNormalization()(c)
    c = ELU()(c)
    c = Dropout(0.2)(c)
    c = Convolution2D(48//nb_filters_reduction_factor, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(c)
    c = BatchNormalization()(c)
    c = ELU()(c)				  
    c = Convolution2D(64//nb_filters_reduction_factor, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(c)
    c = BatchNormalization()(c)
    c = ELU()(c)
    x = merge([a, b, c], mode='concat', concat_axis=-1)
    x = Convolution2D(384//nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='linear',
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    
    x = merge([shortcut, x], mode='sum')
    x = ELU()(x)
    
    return x


def inception_resnet_v2_reduction_A(x):
    a = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid', dim_ordering='tf')(x)
    b = Convolution2D(384//nb_filters_reduction_factor, 3, 3, subsample=(2, 2),
                      init='he_normal', border_mode='valid', dim_ordering='tf')(x)
    b = BatchNormalization()(b)
    b = ELU()(b)				  
    c = Convolution2D(256//nb_filters_reduction_factor, 1, 1, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    c = BatchNormalization()(c)
    c = ELU()(c)
    c = Dropout(0.2)(c)
    c = Convolution2D(256//nb_filters_reduction_factor, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(c)
    c = BatchNormalization()(c)
    c = ELU()(c)				  
    c = Convolution2D(384//nb_filters_reduction_factor, 3, 3, subsample=(2, 2),
                      init='he_normal', border_mode='valid', dim_ordering='tf')(c)
    c = BatchNormalization()(c)
    c = ELU()(c)
    x = merge([a, b, c], mode='concat', concat_axis=-1)
    
    return x
    

def inception_resnet_v2_B(x):
    shortcut = x
    
    a = Convolution2D(192//nb_filters_reduction_factor, 1, 1, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    a = BatchNormalization()(a)
    a = ELU()(a)
    b = Convolution2D(128//nb_filters_reduction_factor, 1, 1, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    b = BatchNormalization()(b)
    b = ELU()(b)
    b = Dropout(0.2)(b)
    b = Convolution2D(160//nb_filters_reduction_factor, 1, 7, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(b)
    b = BatchNormalization()(b)
    b = ELU()(b)				  
    b = Convolution2D(192//nb_filters_reduction_factor, 7, 1, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(b)
    
    x = merge([a, b], mode='concat', concat_axis=-1)
    x = Convolution2D(1154//nb_filters_reduction_factor, 1, 1, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    x = merge([shortcut, x], mode='sum')
    
    x = ELU()(x)
    
    return x


def inception_resnet_v2_reduction_B(x):
    a = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid', dim_ordering='tf')(x)
    b = Convolution2D(256//nb_filters_reduction_factor, 1, 1, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    b = BatchNormalization()(b)
    b = ELU()(b)
    b = Dropout(0.2)(b)
    b = Convolution2D(288//nb_filters_reduction_factor, 3, 3, subsample=(2, 2), 
                      init='he_normal', border_mode='valid', dim_ordering='tf')(b)
    b = BatchNormalization()(b)
    b = ELU()(b)				  
    c = Convolution2D(256//nb_filters_reduction_factor, 1, 1, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    c = BatchNormalization()(c)
    c = ELU()(c)
    c = Dropout(0.2)(c)
    c = Convolution2D(288//nb_filters_reduction_factor, 3, 3, subsample=(2, 2),
                      init='he_normal', border_mode='valid', dim_ordering='tf')(c)
    c = BatchNormalization()(c)
    c = ELU()(c)				  
    d = Convolution2D(256//nb_filters_reduction_factor, 1, 1, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    d = BatchNormalization()(d)
    d = ELU()(d)
    d = Dropout(0.2)(d)
    d = Convolution2D(288//nb_filters_reduction_factor, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(d)
    d = BatchNormalization()(d)
    d = ELU()(d)				  
    d = Convolution2D(320//nb_filters_reduction_factor, 3, 3, subsample=(2, 2),
                      init='he_normal', border_mode='valid', dim_ordering='tf')(d)
    d = ELU()(d)
    x = merge([a, b, c, d], mode='concat', concat_axis=-1)
    
    return x


def inception_resnet_v2_C(x):
    shortcut = x
    
    a = Convolution2D(192//nb_filters_reduction_factor, 1, 1, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    a = BatchNormalization()(a)
    a = ELU()(a)
    b = Convolution2D(192//nb_filters_reduction_factor, 1, 1, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    b = BatchNormalization()(b)
    b = ELU()(b)
    b = Dropout(0.2)(b)
    b = Convolution2D(224//nb_filters_reduction_factor, 1, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(b)
    b = BatchNormalization()(b)
    b = ELU()(b)					  
    b = Convolution2D(256//nb_filters_reduction_factor, 3, 1, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(b)
    b = BatchNormalization()(b)
    b = ELU()(b)					  
    
    x = merge([a, b], mode='concat', concat_axis=-1)
    x = Convolution2D(2048//nb_filters_reduction_factor, 1, 1, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)	
    x = merge([shortcut, x], mode='sum')
    x = ELU()(x)	
    return x


# In the original paper, these are 4, 7, 3, respectively.
num_A_blocks = 2	# This parameter is important.
num_B_blocks = 4 	# This parameter is important.
num_C_blocks = 2	# This parameter is important.

inputs = Input(shape = (32, 32, 3))

x = inception_resnet_v2_stem(inputs)
for i in range(num_A_blocks):
    x = inception_resnet_v2_A(x)
x = inception_resnet_v2_reduction_A(x)
for i in range(num_B_blocks):
    x = inception_resnet_v2_B(x)
x = inception_resnet_v2_reduction_B(x)
for i in range(num_C_blocks):
    x = inception_resnet_v2_C(x)

x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), border_mode='valid', dim_ordering='tf')(x)
x = Dropout(0.5)(x)
x = Flatten()(x)

predictions = Dense(60, activation='softmax')(x)

model = Model(input=inputs, output=predictions)

data_augmentation = True

# Print the model architeture.
model.summary()

# Compile the model.

model.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])


lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]



# Data augmentation.
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'sparse')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'sparse')


# Fit model.

history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=callbacks_list,
        verbose=2,shuffle=False)

# Saving weight.
model.save_weights('output/Cross-Subject/DenseNet-BC-190-40-NTU-CS.h5')

# Plotting confusion matrix.
datagen = ImageDataGenerator(rescale = 1./255)
generator = datagen.flow_from_directory('data/NTU-RGB+D/Cross-Subject/validation',
                                        target_size=(32, 32),
                                        batch_size=1,
                                        class_mode=None,  # Only data, no labels.
                                        shuffle=False)    # Keep data in same order as labels.

# Making predictions on test set.
y_pred = model.predict_generator(generator,15275)
y_pred  = np.argmax(y_pred, axis=-1)

label_map = (train_generator.class_indices)
print(label_map)

# Action label (Ground-truth).
y_true = np.array([0] * 254 + [1] * 254 + [2] * 255 + [3] * 255 + [4] * 256 + [5] * 257 + [6] * 255 + [7] * 255 + [8] * 255 + [9] * 255 + \
                  [10] * 254  + [11] * 254 + [12] * 253 + [13] * 251 + [14] * 256 + [15] * 256 + [16] * 254 + [17] * 254 + [18] * 254 + [19] * 253 + \
                  [20] * 253 + [21] * 254 + [22] * 254 + [23] * 255  + [24] * 254 + [25] * 255 + [26] * 256  + [27] * 255 + [28] * 255  + [29] * 255 + \
                  [30] * 256 + [31] * 256 + [32] * 257 + [33] * 257 + [34] * 256 + [35] * 255 + [36] * 256 + [37] * 256 + [38] * 256 + [39] * 258 + \
                  [40] * 256 + [41] * 256 + [42] * 256 + [43] * 256 + [44] * 256 + [45] * 256 + [46] * 256 + [47] * 255 + [48] * 256 + [49] * 257 + \
                  [50] * 256 + [51] * 256 + [52] * 256 + [53] * 257 + [54] * 251 + [55] * 256 + [56] * 255 + [57] * 256 + [58] * 218 + [59] * 255)

cnf_matrix = confusion_matrix(y_true, y_pred)
#np.set_printoptions(precision=2)
print(confusion_matrix(y_true, y_pred))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,fontsize=8, rotation=0)
    plt.yticks(tick_marks,classes,fontsize=8)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 fontsize=4,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True action')
    plt.xlabel('Predicted action')

# Plot normalized confusion matrix
plt.figure(figsize=(9,9))
plot_confusion_matrix(cnf_matrix, classes = ['Drinking 1', 'Eating 2', 'Brushing teeth 3', 'Brushing hair 4', 'Dropping 5', 'Picking up 6', 'Throwing 7', 'Sitting down 8', 'Standing up 9', 'Clapping 10',
                                             'Reading 11', 'Writing 12', 'Tearing up paper 13', 'Wearing jacket 14', 'Taking off jacket 15', 'Wearing a shoe 16', 'Taking off a shoe 17', 'Wearing on glasses 18',
                                             'Taking off glasses 19', 'Puting on a hat/cap 20', 'Taking off a hat/cap 21', 'Cheering up 22', 'Hand waving 23', 'Kicking something 24', 'Reaching into self pocket 25',
                                             'Hopping 26', 'Jumping up 27', 'Making/Answering a phone call 28', 'Playing with phone 29', 'Pyping 30', 'Pointing to something 31', 'Taking selfie 32', 'Checking time 33',
                                             'Rubbing two hands together 34', 'Bowing 35', 'Shaking head 36', 'Wiping face 37', 'Saluting 38', 'Putting palms together 39', 'Crossing hands in front 40', 'Sneezing/Coughing 41',
                                             'Staggering 42', 'Falling down 43', 'Touching head 44', 'Touching chest 45', 'Touching back 46', 'Touching neck 47', 'Vomiting 48', 'Fanning self 49', 'Punching/Slapping other person 50',
                                             'Kicking other person 51', 'Pushing other person 52', 'Patting others back 53', 'Pointing to the other person 54', 'Hugging 55', 'Giving something to other person 56',
                                             'Touching other persons pocket 57', 'Handshaking 58', 'Walking towards each other 59', 'Walking apart from each other 60'], normalize=True,
                                             title='Confusion Matrix for NTU-RGB+D/Cross-Subject')
                     
plt.savefig('output/Cross-Subject/Confusion-Matrix-DenseNet-BC-190-40-NTU-CS.png')

# List all data in history.
print(history.history.keys())

# Grab the history object dictionary.
H = history.history

last_test_acc = history.history['val_acc'][-1] * 100
last_train_loss = history.history['loss'][-1] 
last_test_acc = round(last_test_acc, 2)
last_train_loss = round(last_train_loss, 6)
train_loss = 'Training Loss, min = ' +  str(last_train_loss)
test_acc = 'Test Accuracy, max = ' + str(last_test_acc) +' (%)'
 
# Plot the training loss and accuracy.
N = np.arange(0, len(H["loss"]))
plt.style.use("ggplot")
plt.figure()
axes = plt.gca()
axes.set_ylim([0.0,2.0])
plt.plot(N, H['loss'],linewidth=2.5,label=train_loss,color='blue')
plt.plot(N, H['val_acc'],linewidth=2.5, label=test_acc,color='red')
plt.title('DenseNet-BC (L=190, k=40) on NTU-RGB+D/Cross-Subject',fontsize=10, fontweight='bold',color = 'Gray')
plt.xlabel('Number of Epochs',fontsize=10, fontweight='bold',color = 'Gray')
plt.ylabel('Training Loss and Test Accuracy',fontsize=10, fontweight='bold',color = 'Gray')
plt.legend()

# Save the figure.
plt.savefig('output/Cross-Subject/DenseNet-BC-190-40-NTU-CS.png')
plt.show()
