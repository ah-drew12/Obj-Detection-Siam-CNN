import tensorflow as tf
from tensorflow import keras
from tensorflow import io
import pandas as pd
import numpy as np
import os
from os import path as osp
import xml.etree.ElementTree as ET
from keras import layers
from keras import initializers
from keras.layers import Conv2D, Dense, Flatten, Input, Dropout, MaxPooling2D,AveragePooling2D,BatchNormalization, DepthwiseConv2D,ReLU,GlobalAveragePooling2D,Reshape,LayerNormalization
from keras.applications import MobileNetV2,ResNet50V2,VGG16,ResNet50
from keras.models import Sequential
import time
import matplotlib.pyplot as plt
import cv2

from keras.models import clone_model,load_model
# import tensorflow_models as tfm
import tensorflow.compat.v1 as teff

#This script executes object localization task that consists in detecting the boundaries of an object. Those bounds will be sent to the classificator to detect the object inside the frame.

teff.enable_eager_execution(teff.ConfigProto(log_device_placement=True))
gpus = tf.config.list_logical_devices('GPU')

tf.config.run_functions_eagerly(True)
image_test=cv2.imread('input_image.jpg')# TEST IMAGE

dataset = tf.data.TFRecordDataset('256_bounding_box_dataset_for_localization_13_frames.tfrecord') # DATASET
print(dataset)

def parse_record(record):
    feature_description = {
        'img': tf.io.FixedLenFeature([], tf.string),
        'cords': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_record = tf.io.parse_single_example(record, feature_description)
    img = tf.io.parse_tensor(parsed_record['img'], out_type=tf.float32)
    cords = tf.io.parse_tensor(parsed_record['cords'], out_type=tf.float32)

    return img, cords


dataset = dataset.map(parse_record)

gpus = tf.config.list_logical_devices('GPU')

def IoU_loss(y, y_pred):

    #LOSS FUNCTION FOR TRAINING
    #This function does not realizes a IOU_loss function in original as it could googled (comparing sizes of frame's area). This method was untrainable because the model trains to find
    #an area size not the area coordinates. So, this loss function evaluates squared error of coordinates of the frame plus the area size of this frame.

    t1 = y
    t2 = y_pred
    rows, columns = t1.get_shape()
    intersection = 0
    union = 0
    area1_sum = 0
    area1_pred_sum = 0
    IoU_sum = 0
    x1_loss = 0
    x2_loss = 0
    y1_loss = 0
    y2_loss = 0
    for i in range(rows):
        min_x_og, min_y_og, max_x_og, max_y_og = tf.split(t1[i], 4, axis=0)
        fact_nump = np.array(t1[i])

        x1_pred, y1_pred, x2_pred, y2_pred = tf.split(t2[0][i], 4, axis=0)
        pred_nump = np.array(t2[0][i])
        detect_arr = np.array([0.81984735, 0.41779888, 0.93320614, 0.75407606])

        if (i == 3 and np.all(np.isclose(fact_nump, detect_arr, atol=1e-5))): print(
            'fact values: ' + str(fact_nump) + '\npred_values: ' + str(pred_nump))



        min_x_pred = x1_pred
        min_y_pred = y1_pred

        max_x_pred = x2_pred
        max_y_pred = y2_pred

        min_x_pred_nump = np.array(min_x_pred)
        min_y_pred_nump = np.array(min_y_pred)
        max_x_pred_nump = np.array(max_x_pred)
        max_y_pred_nump = np.array(max_y_pred)

        x_overlap = tf.maximum(0.0, tf.minimum(max_x_og, max_x_pred) - tf.maximum(min_x_og, min_x_pred))
        y_overlap = tf.maximum(0.0, tf.minimum(max_y_og, max_y_pred) - tf.maximum(min_y_og, min_y_pred))
#
        x_over_nump = np.array(x_overlap)
        y_over_nump = np.array(y_overlap)

        intersection += x_overlap * y_overlap

        area1 = (max_x_og - min_x_og) * (max_y_og - min_y_og)
        area1_pred = (max_x_pred - min_x_pred) * (max_y_pred - min_y_pred)

        area1_nump = np.array(area1)
        area1_pred_nump = np.array(area1_pred)

        area_diff=(area1-area1_pred)**2
        x1_loss += (min_x_og - min_x_pred)** 2
        x2_loss += (max_x_og - max_x_pred)** 2
        y1_loss += (min_y_og - min_y_pred)** 2
        y2_loss += (max_y_og - max_y_pred)** 2

        area1_sum += area1
        area1_pred_sum += area1_pred
        area1_sum_nump = np.array(area1_sum)
        area1_pred_sum_nump = np.array(area1_pred_sum)
        # IoU_sum+=area_diff-intersection
        IoU_sum+= x1_loss+x2_loss+y1_loss+y2_loss
    # (area1 - area1_pred) ** 2 +
    # IoU_sum += intersection / (area1_sum+area1_pred_sum)
    # loss=0.8-IoU
    # loss=(area1_sum-area1_pred_sum)**2
    # loss=x1_loss+x2_loss+y1_loss+y2_loss
    # -intersection**2
    loss = IoU_sum
    loss_numpy = np.array(loss)
    return loss

####################################

class Model(tf.keras.Model):
    #Custom Class of Model
    def __init__(self, nn_box):
        super(Model, self).__init__()
        self.nn_box = nn_box
        self.box_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.9, beta_2=0.9)
        # self.box_optimizer = tf.keras.optimizers.Adamax(0.00005,beta_1=0.9,beta_2=0.9)
        # self.box_optimizer = tf.keras.optimizers.Adam(0.000005,beta_1=0.7,beta_2=0.5)
        # self.box_optimizer = tf.keras.optimizers.SGD(0.001,momentum=0.9)
    @tf.function
    def training_step(self, x, y):
        # x input image
        # y output boxes coords

        with tf.GradientTape() as tape_box:
            # x=tf.image.resize(x,(256,256))

            y_pred = self.nn_box(tf.reshape(x, [-1, 256, 256, 3]), training=True)
            y_pred_numpy = np.array(y_pred)
            y_pred = tf.reshape(y_pred, [-1, 13, 4])

            loss = IoU_loss(y, y_pred)
            loss_nump = np.array(loss)

        grads = tape_box.gradient(loss, self.nn_box.trainable_variables)
        # grads_numpy = np.array(grads[9])
        self.box_optimizer.apply_gradients(zip(grads, self.nn_box.trainable_variables))
        return loss

    def test_step(self, x, y):
        # x = tf.image.resize(x, (256, 256))

        y_pred = self.nn_box(tf.reshape(x, [-1, 256, 256, 3]))
        y_pred_numpy = np.array(y_pred)
        y_pred = tf.reshape(y_pred, [-1, 13, 4])

        val_loss = IoU_loss(y, y_pred)
        loss_nump = np.array(val_loss)
        return val_loss


boxregressor = None

# ##################################################

def get_cnn_block(depth,kernel):

    return Sequential([
        # DepthwiseConv2D(kernel),
        Conv2D(depth,kernel,activation='relu', padding = "same"),
        Conv2D(depth, kernel,activation='relu', padding="same"),
        BatchNormalization(),

    ])

def get_cnn_block_2(depth,kernel):

    return Sequential([
        Conv2D(depth,kernel,activation='relu', padding = "same"),
        # BatchNormalization(),
        Conv2D(depth, kernel,activation='relu', padding="same"),
        # BatchNormalization(),
        Conv2D(depth, kernel,activation='relu', padding="same",strides=2),
        BatchNormalization()
    ])

IMAGE_SIZE=384
ALPHA=1.0
#Model architecture
with tf.device('/GPU:0'):
    input = Input((256, 256, 3),name='input_layer_to_localiz')
    model = keras.Sequential()
    model.add(input)
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(52,activation='relu'))

    boxregressor = tf.keras.Model(inputs=model.input, outputs=model.output)
    # boxregressor = tf.keras.Model(inputs=loaded_model.input,outputs=loaded_model.output)
localizator = Model(boxregressor)
localizator.built = True
boxregressor.summary()
# ##########################################################

images = []
coords = []


def testing():
    #Output image and draw boundaries of object above the image.
    for ii, cc in dataset.take(5):
        # ii=tf.image.resize(ii,(256,256))
        pred = localizator.nn_box(tf.reshape(ii, [-1, 256, 256, 3]))
        image = np.zeros([256, 256, 3], dtype=np.float64)
        i = ii[:, :, :]
        pred = tf.reshape(pred, [-1, 13, 4])
        c = pred[0]
        с_fact = ((np.array(cc)) * 256).astype(dtype=np.float64)
        cc_numpy_x = ((cc.numpy()) * 1310).astype(dtype=np.int16)
        cc_numpy_y = ((cc.numpy()) * 736).astype(dtype=np.int16)
        cc_numpy = ((cc.numpy())*256).astype(dtype=np.int16)

        # ax=plt.subplot(1,1,num+1)
        i = ((i.numpy()) * 255).astype(dtype=np.float64)

        c = ((c.numpy())* 256).astype(dtype=np.int16)

        # for predicted
        color = (255, 255, 0)
        for bb in c:
            i = cv2.rectangle(i, (bb[0], bb[1]), (bb[2], bb[3]), color=(255, 255, 0), thickness=1)
        for bb in cc_numpy:
            i = cv2.rectangle(i, (bb[0], bb[1]), (bb[2], bb[3]), color=(255, 0, 0), thickness=1)

        # plt.figure()
        image = (i) / 255
        image = tf.convert_to_tensor(image, dtype=tf.float64)
        plt.figure()
        plt.imshow(image)

        print('image outputed')



start_time = time.time()
from IPython.display import clear_output

epochs = 200
plt.figure(1)

train_hist = np.array(np.empty([0]))
val_hist = np.array(np.empty([0]))
val_min_loss=0

#Loop for training
#Model is training on the first 8 items in dataset. After the 8th item, it starts to validate the model with outputting a result.
# The training process starts to show images with frames when val_loss less than 27. (RED frames is a needed boundary, yellow is a predicted boundary)
#If val_loss less than 24, the program starts to save weights of the model.

for epoch in range(1, epochs + 1):

    loss = 0
    lc = 0
    val_lc = 0
    val_loss = 0
    print(epoch)
    for step, (i, c) in enumerate(dataset):
        if step == 8: break
        loss += tf.reduce_mean(localizator.training_step(i, c))
        loss_nump = loss.numpy()
        lc += 1

    for step, (i, c) in enumerate(dataset):
        if step < 7:
            continue
        else:
            val_loss += localizator.test_step(i, c)
            val_loss_nump = val_loss.numpy()
            val_lc += 1
    if epoch==1:
        val_min_loss=val_loss_nump
        text='D:/tmp_modified_classificatior/localizator_weights_25_01/1/222/22/223/'+str(epoch)+'_'+str(val_min_loss/val_lc)+'.keras'
        localizator.save_weights(text)
        testing()
    elif val_loss<val_min_loss:
        ratio=val_loss/val_min_loss
        if (
        # ((ratio)<0.9
        #         and
                int(val_loss/val_lc)<27): testing() # зменшити спам фотографіями
        val_min_loss=val_loss_nump
        text='D:/tmp_modified_classificatior/localizator_weights_25_01/1/222/22/223/'+str(epoch)+'_'+str(val_min_loss/val_lc)+'.keras'
        if ((val_min_loss/val_lc)<24): localizator.save_weights(text)

    clear_output(wait=True)

    print('train loss: ' + str((loss / lc).numpy()) + ' , val_loss: ' + str((val_loss / val_lc).numpy()))

    train_hist = np.append(train_hist, loss / lc)

    val_hist = np.append(val_hist, val_loss / val_lc)
    # if epoch!= 1:
    plt.figure(1)
    plt.plot(np.arange(0, len(train_hist)), train_hist,'b')
    plt.plot(np.arange(0, len(val_hist)), val_hist,'m')
    plt.legend(['train_loss','val_loss'], loc="upper left")



    plt.show()
end_time = time.time() - start_time #Training time
print(end_time)

(pd.DataFrame(train_hist)).to_csv('D:/tmp_modified_classificatior/localizator_weights_25_01/1/222/22/223/train_hist.csv')
(pd.DataFrame(val_hist)).to_csv('D:/tmp_modified_classificatior/localizator_weights_25_01/1/222/22/223/val_hist.csv')

localizator.nn_box.save('localizator_model_name.h5')

print('d')
