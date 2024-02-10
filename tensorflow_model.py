import cv2
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import Sequential, Model, Input
from keras.models import clone_model,load_model
from keras.layers import GlobalAveragePooling2D, Dense, Reshape, Conv2D, BatchNormalization, Concatenate, MaxPooling2D, PReLU,ReLU,Flatten,Dropout
from sklearn.model_selection import train_test_split
import itertools
import  random
tf.config.run_functions_eagerly(True)
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

    #Select images for pairing in future

    x_train, x_test, y_train, y_test = train_test_split(images,labels,test_size=test_size,random_state=42)
    if get_another_sample is False:

        x_train=np.stack(np.array(x_train))
        x_test=np.stack(np.array(x_test))
        y_train=np.stack(np.array(y_train))
        y_test=np.stack(np.array(y_test))

        random_indexes = np.random.choice(x_train.shape[0], 100, replace=False)

        x_train_sample, y_train_sample = x_train[random_indexes], y_train[random_indexes]

        random_indexes = np.random.choice(x_test.shape[0], 25, replace=False)

        x_test_sample, y_test_sample = x_test[random_indexes], y_test[random_indexes]
        return x_train_sample, y_train_sample, x_test_sample, y_test_sample
    else:
        random_indexes = np.random.choice(x_train.shape[0], 100, replace=False)

        x_train_sample, y_train_sample = x_train[random_indexes], y_train[random_indexes]

        random_indexes = np.random.choice(x_test.shape[0], 25, replace=False)

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
    #After those manipulation we reduce our dataset from 45150 pairs to 2700.
    #For creating an example of False similarity (pair of different persons), we use condition that randomly generated number from 0 to 1 will be bigger than 0.9.
    #That condition helps us to use different photos for False similarity. If we set just 6 photos without the condition, it will be taking the same first 6 photos each time and model could be overfitted for that photos.


    #variables for negative example for each photo
    temp_t0=np.zeros((128,128,3))

    count_unknown_paired=0#count of false similarity
    count_label_paired=0#count of true similarity
    sum=0
    for t in itertools.product(tuples,tuples):


        pair_A,pair_B=t
        A_index,img_A,label_A=t[0]
        B_index,img_B,label_B=t[1]
        # if (B_index>=A_index):
        #     sum += 1
        #     continue

        new_label = np.zeros(2)
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

        # if labels is equal then we setting 1 that's means the image label matching

        if (int(label_A==label_B)):
            if(count_label_paired<3):


                new_label = [1, 0]
                count_label_paired+=1
            else: continue
        else:
            if (count_unknown_paired < 6):
                rand=random.random()
                if (rand>0.9):
                    #unknown

                    new_label = [0,1]
                    count_unknown_paired+=1
                else: continue
            else: continue

        x_pairs.append([img_A,img_B])
        y_pairs.append(new_label)
        count+=1

    x_pairs=np.array(x_pairs)
    y_pairs=np.array(y_pairs)

    return x_pairs, y_pairs

#Selecting images for model training
x_train_sample,y_train_sample,x_test_sample,y_test_sample=get_datasets_for_loop(images,labels,0.2)

#Creating paired dataset with selected images and labeling it with True or False (person on images are similar or not )
x_train_pairs, y_train_pairs = make_paired_dataset(x_train_sample,y_train_sample)
x_test_pairs, y_test_pairs = make_paired_dataset(x_test_sample,y_test_sample)



#Creating a model
img_A_inp=Input(shape=(128,128,3),name="img_A_inp")
img_B_inp=Input(shape=(128,128,3),name="img_B_inp")

def get_cnn_block(depth,kernel):
    return Sequential([
        Conv2D(depth,kernel,data_format='channels_last',activation = 'relu', padding = "same"),
        # BatchNormalization(input_shape=(128,128,3,1),batch_input_shape=(None,32)),
        ReLU()
    ])

DEPTH=64

cnn=Sequential([
        Reshape(target_shape=(128,128,3)),
        Conv2D(64, 10, activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

    Conv2D(128, 10, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    Dropout(0.25),
    Conv2D(256, 10, activation='relu'),
    Conv2D(256, 4, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.3)

])

feature_vector_A=cnn(img_A_inp)
feature_vector_B=cnn(img_B_inp)


concat= Concatenate()([feature_vector_A,feature_vector_B])

dense=Dense(64, activation='relu',name='siamese_dense_pre_output_layer3')(concat)

output=Dense(2,activation='softmax',name='siamese_output_layer')(dense)

siamese_cnn=Model(inputs=[img_A_inp,img_B_inp], outputs=output)

siamese_cnn.summary()




opt = keras.optimizers.Adam(learning_rate=0.00005,
                                    beta_1=0.8,
                                    beta_2=0.8)

#Creating a custom metrics that work similar to keras metrics FalseNegative,TruePositive, FalsePositive,TrueNegative
class false_neg(keras.metrics.Metric):
    def __init__(self, name = 'false_negative_metric', **kwargs):
        super(false_neg, self).__init__(**kwargs)
        self.false_neg = self.add_weight('false_neg', initializer = 'zeros')

    @tf.function
    def update_state(self, y_true, y_pred,sample_weight=None):
        y_pred_numpy = y_pred.numpy()
        y_pred_numpy = [[1 if pred == np.max(y_pred_temp) else 0 for pred in y_pred_temp] for y_pred_temp in y_pred_numpy[:]]
        y_true_numpy = y_true.numpy()
        y_true_numpy = [[1 if true == np.max(y_test_temp) else 0 for true in y_test_temp] for y_test_temp in y_true_numpy[:]]


        a=1
        for pred, true in zip(y_pred_numpy, y_true_numpy):
            if ((pred != true) & (pred[1] == 1)):
                self.false_neg.assign_add(1)

    def reset_state(self):
        self.false_neg.assign(0)

    def result(self):
        return self.false_neg


class true_pos(keras.metrics.Metric):
    def __init__(self, name = 'true_positive_metric', **kwargs):
        super(true_pos, self).__init__(**kwargs)
        self.true_pos = self.add_weight('true_pos', initializer = 'zeros')

    @tf.function
    def update_state(self, y_true, y_pred,sample_weight=None):
        y_pred_numpy = y_pred.numpy()
        y_pred_numpy = [[1 if pred == np.max(y_pred_temp) else 0 for pred in y_pred_temp] for y_pred_temp in y_pred_numpy[:]]
        y_true_numpy = y_true.numpy()
        y_true_numpy = [[1 if true == np.max(y_test_temp) else 0 for true in y_test_temp] for y_test_temp in y_true_numpy[:]]

        for pred, true in zip(y_pred_numpy, y_true_numpy):
            if ((pred == true)&(pred[0] == 1)):
                self.true_pos.assign_add(1)

    def reset_state(self):
        self.true_pos.assign(0)

    def result(self):
        return self.true_pos

class true_neg(keras.metrics.Metric):
    def __init__(self, name = 'true_positive_metric', **kwargs):
        super(true_neg, self).__init__(**kwargs)
        self.true_neg = self.add_weight('true_neg', initializer = 'zeros')

    @tf.function
    def update_state(self, y_true, y_pred,sample_weight=None):
        y_pred_numpy = y_pred.numpy()
        y_pred_numpy = [[1 if pred == np.max(y_pred_temp) else 0 for pred in y_pred_temp] for y_pred_temp in
                        y_pred_numpy[:]]
        y_true_numpy = y_true.numpy()
        y_true_numpy = [[1 if true == np.max(y_test_temp) else 0 for true in y_test_temp] for y_test_temp in
                        y_true_numpy[:]]

        for pred, true in zip(y_pred_numpy, y_true_numpy):
            if ((pred == true) & (pred[1] == 1)):
                self.true_neg.assign_add(1)

    def reset_state(self):
        self.true_neg.assign(0)

    def result(self):
        return self.true_neg

class false_pos(keras.metrics.Metric):
    def __init__(self, name = 'true_positive_metric', **kwargs):
        super(false_pos, self).__init__(**kwargs)
        self.false_pos = self.add_weight('false_pos', initializer = 'zeros')

    @tf.function
    def update_state(self, y_true, y_pred,sample_weight=None):
        y_pred_numpy = y_pred.numpy()
        y_pred_numpy = [[1 if pred == np.max(y_pred_temp) else 0 for pred in y_pred_temp] for y_pred_temp in
                        y_pred_numpy[:]]
        y_true_numpy = y_true.numpy()
        y_true_numpy = [[1 if true == np.max(y_test_temp) else 0 for true in y_test_temp] for y_test_temp in
                        y_true_numpy[:]]

        for pred, true in zip(y_pred_numpy, y_true_numpy):
            if ((pred != true) & (pred[0] == 1)):
                self.false_pos.assign_add(1)

    def reset_state(self):
        self.false_pos.assign(0)

    def result(self):
        return self.false_pos


#Custom metric that sums FalseNegatives and FalsePositives
class comb_fals(keras.metrics.Metric):
    def __init__(self, name = 'true_positive_metric', **kwargs):
        super(comb_fals, self).__init__(**kwargs)
        self.comb_fals = self.add_weight('comb_fals', initializer = 'zeros')

    @tf.function
    def update_state(self, y_true, y_pred,sample_weight=None):
        y_pred_numpy = y_pred.numpy()
        y_pred_numpy = [[1 if pred == np.max(y_pred_temp) else 0 for pred in y_pred_temp] for y_pred_temp in
                        y_pred_numpy[:]]
        y_true_numpy = y_true.numpy()
        y_true_numpy = [[1 if true == np.max(y_test_temp) else 0 for true in y_test_temp] for y_test_temp in
                        y_true_numpy[:]]
        for pred, true in zip(y_pred_numpy, y_true_numpy):
            #fals negative
            if ((pred != true) & (pred[1] == 1)):
                self.comb_fals.assign_add(1)

                #fals positive
            if ((pred != true) & (pred[0] == 1)):
                self.comb_fals.assign_add(1)


    def reset_state(self):
        self.comb_fals.assign(0)



    def result(self):
        return self.comb_fals



siamese_cnn.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy',comb_fals(),false_neg(),true_pos(),true_neg(),false_pos()])


mc = tf.keras.callbacks.ModelCheckpoint('D:/tmp/model_13/1_2234523_1321112/classificator_12_photos/{epoch:02d}__{val_comb_fals:02f}.keras', monitor='val_comb_fals', mode='min', save_best_only=True, verbose=1)

history = siamese_cnn.fit(x=[x_train_pairs[:,0],x_train_pairs[:,1]], y=y_train_pairs, validation_data=[[x_test_pairs[:,0],x_test_pairs[:,1]], y_test_pairs],
                                                    epochs=30, batch_size=5,callbacks=[mc])


siamese_cnn.save('model_siamese_cnn_name.h5')
plt.figure()
plt.title('loss graph')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()
plt.figure()
plt.title('accuracy graph')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()
