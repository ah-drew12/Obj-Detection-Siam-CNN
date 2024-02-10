
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import models
from keras import Sequential, Model, Input
from keras.models import clone_model,load_model
from keras.layers import GlobalAveragePooling2D, Dense, Reshape, Conv2D, BatchNormalization, Concatenate, MaxPooling2D, PReLU,ReLU,Flatten,Dropout,DepthwiseConv2D
from keras.metrics import TruePositives,TrueNegatives ,FalsePositives ,FalseNegatives, CategoricalAccuracy
from sklearn.model_selection import train_test_split
from keras import backend as K
import numpy as np
from keras import initializers

#This program creates a classificator for 12 classes.
# Classificator have to identify "The Office" characters and other people that label as unknown

tf.config.run_functions_eagerly(True)
# classification_dataset_v1_inside_modified_siamese_cnn
dataset = tf.data.TFRecordDataset('classification_dataset_modified_siamese_cnn_256pix.tfrecord')
# dataset = tf.data.TFRecordDataset('classification_the_office_dataset.tfrecord')
#
def parse_record(record):
    images = []#
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

def create_embedding(images,labels):
#Creating one hot encoding for images
    x, y = [], []
    tuples = [(x1, y1) for x1, y1 in zip(images, labels)]
    count = 0
    embedding = ['angela_martin','creed_bratton', 'dwight_shrute', 'jim_halpert', 'kelly_kapoor', 'kevin_malone',
                 'meredith_palmer' ,'michael_scott', 'oscar_martinez' ,'pam_beesly', 'stanley_hudson', 'unknown']

    for t in tuples:
        new_label = np.zeros(12)
        label=(t[1].numpy()).decode("utf-8")
        if label == 'angela_martin': new_label[0] = 1
        if label == 'creed_bratton': new_label[1] = 1
        if label == 'dwight_shrute': new_label[2] = 1
        if label == 'jim_halpert': new_label[3] = 1
        if label == 'kelly_kapoor': new_label[4] = 1
        if label == 'kevin_malone': new_label[5] = 1
        if label == 'meredith_palmer': new_label[6] = 1
        if label == 'michael_scott': new_label[7] = 1
        if label == 'oscar_martinez': new_label[8] = 1
        if label == 'pam_beesly': new_label[9] = 1
        if label == 'stanley_hudson': new_label[10] = 1
        if label == 'unknown': new_label[11] = 1
        y.append(new_label)
    return y,embedding

y_embed,embedding=create_embedding(images,labels)
# data = preprocessing.scale(images.reshape(images.shape[0], -1))
x_train, x_test, y_train, y_test = train_test_split(images,y_embed,test_size=0.3)


#Model creating
cnn=Sequential([
    Conv2D(32,3,name='classification_conv_layer_2'),
    PReLU(),
    MaxPooling2D(2,name='classification_maxpool_layer_3'),

    BatchNormalization(),

    Conv2D(64,3,name='classification_conv_layer_4'),
    PReLU(),
    MaxPooling2D(2,name='classification_maxpool_layer_5'),

    Conv2D(128, 3, name='classification_conv_layer_41'),
    PReLU( ),
    MaxPooling2D(2, name='classification_maxpool_layer_51'),
    Conv2D(256,3,name='classification_conv_layer_6'),
    ReLU(),
    MaxPooling2D(2,name='classification_maxpool_layer_7'),
    Conv2D(512,3,name='classification_conv_layer_8'),
    ReLU(),
    MaxPooling2D(2,name='classification_maxpool_layer_9'),
    Conv2D(1024, 3, name='classification_conv_layer_10'),
    PReLU(),
    MaxPooling2D(2, name='classification_maxpool_layer_11'),
    #
    Flatten(),


])
def get_cnn_block(depth,kernel):

    return Sequential([

        Conv2D(depth,kernel, padding = "same"),
        Conv2D(depth, kernel, padding="same",strides=2),
        BatchNormalization(),
        MaxPooling2D()
    ])
initializer_1=initializers.RandomNormal(stddev=2)

input_1=tf.keras.Input(shape=(256, 256, 3),name='input_1_classificator_of_embedd')
x=get_cnn_block(32,5)(input_1)
x=get_cnn_block(64,5)(x)
x=get_cnn_block(128,5)(x)
# x=get_cnn_block(512,1)(x)
x=Flatten()(x)
x = Dropout(0.3)(x)
x=Dense(128,activation='relu')(x)
initializer = keras.initializers.RandomUniform(minval=0.0, maxval=5.0)
output=Dense(12,activation='softmax',use_bias=True, name='classifcation_output_layer7',kernel_initializer=initializer)(x)

classification_for_office=Model(inputs=input_1, outputs=output)


classification_for_office.summary()

opt = keras.optimizers.Adam(learning_rate=0.0005,
                                    beta_1=0.6,
                                    beta_2=0.7,
                                    # epsilon=temp_epsilon,amsgrad=temp_amsgrad
                                )

opt_2 = keras.optimizers.Adam(learning_rate=0.00005,
                                    beta_1=0.9,
                                    beta_2=0.9,
                                )

opt_3 = keras.optimizers.Adamax(learning_rate=0.005,
                                    beta_1=0.4,
                                    beta_2=0.8,
                                )


classification_for_office.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['categorical_accuracy'])
# classification_for_office=load_model('classificator_12_embed_for_siamese_cnn.h5')

pred=classification_for_office.predict(images)
# pred=classification_for_office.evaluate(x=images,y=np.array(y_embed))
mc = tf.keras.callbacks.ModelCheckpoint('D:/tmp_classificator_embed_for_siamese_cnn/model_55/1/{epoch:02d}__{val_categorical_accuracy:02f}.keras', monitor='val_categorical_accuracy', mode='max', save_best_only=True, verbose=1)
# classification_for_office.load_weights('D:/tmp_classificator_embed_for_siamese_cnn/34__0.653846.keras')

history = classification_for_office.fit(x=x_train, y=y_train,validation_data=(x_test,y_test),
                                                    epochs=200, batch_size=32,
                    callbacks=[mc]
                    )

#After checking prediction on all saved results, in the end we get model with the best result
import os
PATH='D:/tmp_classificator_embed_for_siamese_cnn/model_55/1/'
dir_list=os.listdir(PATH)
for model_ver in dir_list:
    path = PATH + model_ver
    classification_for_office.load_weights(path)

    pred_1 = classification_for_office.predict(x_train)
    pred_1_log = classification_for_office.evaluate(x_train,y=y_train)
    pred_2 = classification_for_office.predict(x_test)
    pred_2_log = classification_for_office.evaluate(x_test,y=y_test)


#Model saving in format .h5
classification_for_office.save('classificator_12_embed_for_siamese_cnn.h5')

#Prediction on test dataset with output on plot
for image in x_test:
    pred=classification_for_office.predict(image.reshape(-1,256,256,3))
    plt.imshow(image)
    plt.show()
pred=classification_for_office.predict(x_test)

for image in x_test:
    pred=classification_for_office.predict(image.reshape(-1,256,256,3))
    pred = [1 if el==np.max(pred) else 0 for el in pred[0,:]]

    plt.imshow(image)
    if pred[0] == 1 : label = 'angela_martin'
    if  pred[1] == 1: label = 'creed_bratton'
    if pred[2] == 1 : label = 'dwight_shrute'
    if  pred[3] == 1: label = 'jim_halpert'
    if  pred[4] == 1:  label = 'kelly_kapoor'
    if  pred[5] == 1: label = 'kevin_malone'
    if  pred[6] == 1: label = 'meredith_palmer'
    if  pred[7] == 1: label = 'michael_scott'
    if  pred[8] == 1: label = 'oscar_martinez'
    if  pred[9] == 1:  label = 'pam_beesly'
    if  pred[10] == 1: label = 'stanley_hudson'
    if  pred[11] == 1: label = 'unknown'
    plt.title(str(label))
    plt.show()

print('model training is finished')
