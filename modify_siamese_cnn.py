import random

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import models
from keras import Sequential, Model, Input
from keras.models import clone_model,load_model
from keras.layers import GlobalAveragePooling2D, Dense, Reshape, Conv2D, BatchNormalization, Concatenate, MaxPooling2D, PReLU,ReLU,Flatten,Dropout, Resizing
from keras.metrics import TruePositives,TrueNegatives ,FalsePositives ,FalseNegatives
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras import initializers
from IPython.display import clear_output
import itertools
import tensorflow.compat.v1 as teff
import time
teff.enable_eager_execution(teff.ConfigProto(log_device_placement=True))
gpus = tf.config.list_logical_devices('GPU')
tf.config.run_functions_eagerly(True)

#This script executes an extension of output for labeling frames that would get from localization model (Object Detection Task).
#The given task is requires to classify frames with Siamese CNN.
#So we realized classic Siamese CNN that compares 2 image by sending it through two same branches and then concatenating results and defining whether it is similar or not.
#
# dataset = tf.data.TFRecordDataset('classification_the_office_dataset.tfrecord')
dataset = tf.data.TFRecordDataset('classification_dataset_modified_siamese_cnn_256pix.tfrecord')
def parse_record(record):
    images = []
    labels = []
    #read data from tfrecord file
    for raw_record in dataset.as_numpy_iterator():
        feature_description = {
            'img': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string)
        }
        parsed_record = tf.io.parse_single_example(raw_record, feature_description)
        img = tf.io.parse_tensor(parsed_record['img'], out_type=tf.float32)
        label = tf.io.parse_tensor(parsed_record['label'], out_type=tf.string)

        images.append(tf.image.resize(img, (128, 128)))
        labels.append(label)
        print(img)
        print(label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels


images, labels = parse_record(dataset)

def get_datasets_for_loop(images,labels,test_size,get_another_sample=False):

    #Divide dataset into train/test and select images from those sets for work.

    x_train, x_test, y_train, y_test = train_test_split(images,labels,test_size=test_size,random_state=42)
    if get_another_sample is False:

        x_train=np.stack(np.array(x_train))
        x_test=np.stack(np.array(x_test))
        y_train=np.stack(np.array(y_train))
        y_test=np.stack(np.array(y_test))
        random_indexes = np.random.choice(x_train.shape[0], 75, replace=False)

        x_train_sample, y_train_sample = x_train[random_indexes], y_train[random_indexes]

        random_indexes = np.random.choice(x_test.shape[0], 50, replace=False)

        x_test_sample, y_test_sample = x_test[random_indexes], y_test[random_indexes]
        return x_train_sample, y_train_sample, x_test_sample, y_test_sample
    else:
        random_indexes = np.random.choice(x_train.shape[0], 70, replace=False)

        x_train_sample, y_train_sample = x_train[random_indexes], y_train[random_indexes]

        random_indexes = np.random.choice(x_test.shape[0], 50, replace=False)

        x_test_sample, y_test_sample = x_test[random_indexes], y_test[random_indexes]
        return x_train_sample, y_train_sample, x_test_sample, y_test_sample

def make_paired_dataset(x,y):
    #Making selected images paired. Result of pairing is 0 or 1 (photos of the same person or not).

    x_pairs, y_pairs = [],[]
    tuples=[(x1,x1_index,y1) for (x1,x1_index),y1 in zip(enumerate(x),y)]
    count=0


    # Function using only 3 image pairs of the same person and 6 - with different person. This limit was selected to expand the variety of images. This helps us to use more data.
    # For example, if you have 300 photos it will be 45150 pairs, because each photo compared to each other (N*(N+1)/2). So much data will cause problems with performance.
    # But when you set a limit to 6 photos it helps us to reduce size of used memory (300*6). The same logic used for photos of the same person (function takes only 3 photos of the same person).
    # After those manipulation we reduce our dataset from 45150 pairs to 2700.
    # For creating an example of False similarity (pair of different persons), we use condition that randomly generated number from 0 to 1 will be bigger than 0.9.
    # That condition helps us to use different photos for False similarity. If we set just 6 photos without the condition, it will be taking the same first 6 photos each time and model could be overfitted for that photos.


    temp_t0=np.zeros((128,128,3))

    count_unknown_paired=0#count of false similarity
    count_label_paired=0#count of true similarity

    for t in itertools.product(tuples,tuples):
        pair_A, pair_B = t
        A_index, img_A, label_A = t[0]
        B_index, img_B, label_B = t[1]
        # if (B_index>=A_index):
        #     sum += 1
        #     continue

       #code for variables for count a negative example for each photo
        if (temp_t0!=np.zeros((128,128,3))).any():
            if ((img_A==temp_t0).all()):
                if((count_unknown_paired==3)&(count_label_paired==3)):
                    continue
                else:pass
        else: temp_t0 = img_A
        if ((img_A!=temp_t0).any()):
            count_unknown_paired = 0
            count_label_paired = 0
            temp_t0 = img_A



        label_A=label_A.decode("utf-8")
        label_B=label_B.decode("utf-8")
        new_label=np.zeros(13)

        # if labels is equal then we setting 1 that's means the image label matching


        #We set an Y array with size of 13 (Not 12 such as classificator where is already exist a class "unknown")
        # to separate parameter "UNKNOWN" from "Unmatching" for the images that classificator could classify with 12 group,
        # but pre-trained siamese-CNN could label that pair of images a non-similar. So in theory , label "unmatched" has to be used only for cases when images in pair non-similar.


        if (int(label_A==label_B)&(label_A !='unknown')):
            if(count_label_paired<3):
                if label_A=='angela_martin': new_label[0]=1
                if label_A == 'creed_bratton': new_label[1]=1
                if label_A == 'dwight_shrute': new_label[2]=1
                if label_A == 'jim_halpert': new_label[3]=1
                if label_A == 'kelly_kapoor': new_label[4]=1
                if label_A == 'kevin_malone': new_label[5]=1
                if label_A == 'meredith_palmer': new_label[6]=1
                if label_A == 'michael_scott': new_label[7]=1
                if label_A == 'oscar_martinez': new_label[8]=1
                if label_A == 'pam_beesly': new_label[9]=1
                if label_A == 'stanley_hudson': new_label[10]=1
                if label_A == 'unknown': new_label[11] = 1
                count_label_paired+=1
            else: continue
        else:

            if (count_unknown_paired < 5):
                rand=random.random()
                if (rand>0.9):
                    #Photo is not similar
                    new_label[12] = 1
                    count_unknown_paired+=1
                else: continue
            else: continue


        x_pairs.append([img_A,img_B])
        y_pairs.append(new_label)
        count+=1
    x_pairs=np.array(x_pairs)
    y_pairs=np.array(y_pairs)
    embedding=['angela_martin' 'creed_bratton' 'dwight_shrute' 'jim_halpert' 'kelly_kapoor' 'kevin_malone'
              'meredith_palmer' 'michael_scott' 'oscar_martinez' 'pam_beesly' 'stanley_hudson','unknown']
    return x_pairs, y_pairs,embedding


def make_paired_dataset_For_siamese(x,y):
    #This function similar in meaning to make_paired_dataset.py , but it does not have a limit of number photo to each group (Groups of similar and non-similar).
    #So this function realise a computation of N*N pairs (for example N=300, therefore output X_DATASET will have 90000 pairs).
    x_pairs, y_pairs = [],[]

    tuples=[(x1,y1) for x1,y1 in zip(x,y)]
    count=0
    for t in itertools.product(tuples,tuples):

        pair_A,pair_B=t
        img_A,label_A=t[0]
        img_B,label_B=t[1]

        new_label=np.zeros(13)
        if int(label_A==label_B):
            if label_A=='angela_martin': new_label[0]=1
            if label_A == 'creed_bratton': new_label[1]=1
            if label_A == 'dwight_shrute': new_label[2]=1
            if label_A == 'jim_halpert': new_label[3]=1
            if label_A == 'kelly_kapoor': new_label[4]=1
            if label_A == 'kevin_malone': new_label[5]=1
            if label_A == 'meredith_palmer': new_label[6]=1
            if label_A == 'michael_scott': new_label[7]=1
            if label_A == 'oscar_martinez': new_label[8]=1
            if label_A == 'pam_beesly': new_label[9]=1
            if label_A == 'stanley_hudson': new_label[10]=1
            if label_A == 'unknown': new_label[11] = 1
        else:
            #not mathcing
            new_label[12] = 1
        x_pairs.append([img_A,img_B])
        y_pairs.append(new_label)
        count+=1
    x_pairs=np.array(x_pairs)
    y_pairs=np.array(y_pairs)

    return x_pairs, y_pairs

x_train_sample,y_train_sample,x_test_sample,y_test_sample=get_datasets_for_loop(images,labels,0.4)
x_train_pairs, y_train_pairs,embedding = make_paired_dataset(x_train_sample,y_train_sample)
x_test_pairs, y_test_pairs,embedding = make_paired_dataset(x_test_sample,y_test_sample)

train_dataset=tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(x_train_pairs),tf.convert_to_tensor(y_train_pairs)),'training_dataset_batched')
test_dataset=tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(x_test_pairs),tf.convert_to_tensor(y_test_pairs)),'testing_dataset_batched')


#model architure


siamese_cnn=load_model('siamese_cnn_model.h5')
siamese_cnn._name="siamese_cnn"
# predd=siamese_cnn.predict([x_train_pairs[:,0],x_train_pairs[:,1]])
siamese_cnn.trainable=False
classificator_for_office=load_model('classificator_12_embed_for_siamese_cnn.h5')
classificator_for_office._name ="classificator_for_scnn"
classificator_for_office.trainable=False

input_1 = tf.keras.Input(shape=(256, 256, 3),name='input_1_out_of')
input_2 = tf.keras.Input(shape=(256, 256, 3),name='input_2_out_of')
# input_1=keras.layers.LayerNormalization(name='normalization_input_1_s')(input_1)
# input_2=keras.layers.LayerNormalization(name='normalization_input_2_s')(input_2)

classificator_office_input=input_1
classificator_officer_output=classificator_for_office(classificator_office_input)
siamese_input = ([Resizing(height=128,width=128,trainable=False)(input_1),Resizing(height=128,width=128,trainable=False)(input_2)])
siamese_output= siamese_cnn(siamese_input)

concatenating_output=Concatenate(axis=1)([classificator_officer_output,siamese_output])
# dense=keras.layers.BatchNormalization(name='normalization_after_concat1_s')(concatenating_output)
initializer=initializers.RandomNormal(stddev=3)
initializer_1=initializers.RandomNormal(stddev=2)
dense=Dense(32,name='dense_10_out_of_model',activation='relu')(concatenating_output)
dense=Dropout(0.25)(dense)

output=Dense(13,activation='softmax',name='output_from_model')(dense)
modified_siamese_cnn = tf.keras.Model([input_1,input_2], output)


class Model(tf.keras.Model):

    #Realizing a model with custom Class Model

    def __init__(self, nn_box):
        super(Model, self).__init__()
        self.nn_box = nn_box
        self.box_optimizer =tf.keras.optimizers.Adam(0.0001, beta_1=0.9, beta_2=0.9)
            # keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
        # tf.keras.optimizers.Adam(0.0001, beta_1=0.9, beta_2=0.9)

    @tf.function
    def training_step(self, x, y):
        # x input image
        # y output boxes coords

        with tf.GradientTape() as tape_box:
            y_pred = self.nn_box([tf.reshape(x[0], [-1, 256, 256, 3]),tf.reshape(x[1], [-1, 256, 256, 3])])
            y_pred_numpy = np.array(y_pred)
            # categorical loss
            loss =tf.keras.losses.CategoricalCrossentropy()(y, y_pred)
            loss_nump = np.array(loss)

        y_pred = self.nn_box([tf.reshape(x[0], [-1, 256, 256, 3]),tf.reshape(x[1], [-1, 256, 256, 3])])
        y_pred_nump = np.array(y_pred)
        grads = tape_box.gradient(loss, self.nn_box.trainable_variables)
        grads_numpy = np.array(grads[9])
        self.box_optimizer.apply_gradients(zip(grads, self.nn_box.trainable_variables))
        return loss

    def test_step(self, x, y):
        y_pred = self.nn_box([tf.reshape(x[0], [-1, 256, 256, 3]),tf.reshape(x[1], [-1, 256, 256, 3])])
        y_pred_numpy = np.array(y_pred)
        #categorical loss
        val_loss =tf.keras.losses.CategoricalCrossentropy()(y.reshape((1,12)), y_pred)
        loss_nump = np.array(val_loss)
        return val_loss
    def predict(self, x):
        y_pred = self.nn_box([tf.reshape(x[0], [-1, 256, 256, 3]),tf.reshape(x[1], [-1, 256, 256, 3])])
        y_pred_numpy = np.array(y_pred)
        #categorical loss

        return y_pred


opt = keras.optimizers.SGD(learning_rate=0.00005,
                                    # beta_1=0.8,
                                    # beta_2=0.7
                           momentum=0.7,
                                    # epsilon=temp_epsilon,amsgrad=temp_amsgrad
                                )
opt_1 = keras.optimizers.Adam(learning_rate=0.00005,
                                    beta_1=0.8,
                                    beta_2=0.9,
                                    # epsilon=temp_epsilon,amsgrad=temp_amsgrad
                                )
opt_2 = keras.optimizers.Adam(learning_rate=0.0005,
                                    beta_1=0.9,
                                    beta_2=0.9,
                                    # epsilon=temp_epsilon,amsgrad=temp_amsgrad
                                )


# modified_siamese_cnn.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['categorical_accuracy'])
opt_3 = keras.optimizers.SGD(learning_rate=0.1)
opt_4 = keras.optimizers.SGD(learning_rate=0.001)


######   TRAINING         ######
mc = tf.keras.callbacks.ModelCheckpoint('D:/tmp_modified_classificatior/model_2222_1/2/{epoch:02d}__{val_categorical_accuracy:02f}.keras', monitor='val_categorical_accuracy', mode='max', save_best_only=True, verbose=1)

modified_siamese_cnn.compile(loss='categorical_crossentropy',optimizer=opt_1,metrics=['categorical_accuracy'])

history = modified_siamese_cnn.fit(x=[x_train_pairs[:,0],x_train_pairs[:,1]], y=y_train_pairs, validation_data=[[x_test_pairs[:,0],x_test_pairs[:,1]], y_test_pairs],
                                                    epochs=30, batch_size=10,callbacks=[mc])
###############################

modified_siamese_cnn.save('modified_siamese_cnn_name.h5')
##### TEST MODULE
pred_test=modified_siamese_cnn.predict([x_test_pairs[:,0],x_test_pairs[:,1]])
pred_train=modified_siamese_cnn.predict([x_train_pairs[:,0],x_train_pairs[:,1]])

pred_test_eval=modified_siamese_cnn.evaluate([x_test_pairs[:,0],x_test_pairs[:,1]],y=y_test_pairs)
pred_train_eval=modified_siamese_cnn.evaluate([x_train_pairs[:,0],x_train_pairs[:,1]],y=y_train_pairs)
(pd.DataFrame(pred_test)).to_csv('modified_siam_cnn_pred_test.csv')
(pd.DataFrame(pred_train)).to_csv('modified_siam_cnn_pred_train.csv')

print('model training is finished')
