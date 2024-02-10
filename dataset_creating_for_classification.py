import tensorflow as tf
from tensorflow import io
import pandas as pd
import numpy as np
import os
from os import path as osp
import xml.etree.ElementTree as ET
import cv2

writer = tf.io.TFRecordWriter('classification_the_office_dataset.tfrecord')

def load_img(img):
    img=tf.io.read_file(img)
    img=tf.image.decode_jpeg(img,channels=3)
    img=tf.cast(img,tf.float32)/255
    img=tf.image.resize(img,(128,128))
    return img


#coords for all test images
cords=[]
path='samples'
folders=os.listdir(path)
dim=(128,128)
for folder in folders:
    images = os.listdir(path+'/'+str(folder))
    # images = os.listdir(path)
    for image in images:
        img_pixels=cv2.imread(path + '/'+str(folder)+'/' + str(image))
        img_pixels=load_img(path + '/'+str(folder)+'/' + str(image))

        # image_to_show=img_pixels.copy()

        # image_to_show[:, :, 0] = img_pixels[:, :, 2]
        # image_to_show[:, :, 2] =img_pixels[:, :, 0]
        # img_pixels=image_to_show

        serialized_img = tf.io.serialize_tensor(img_pixels).numpy()
        serialized_label = tf.io.serialize_tensor(folder).numpy()

        example = tf.train.Example(features=tf.train.Features(feature={
            'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_img])),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_label]))
        }))


        writer.write(example.SerializeToString())

writer.close()


dataset=tf.data.TFRecordDataset('classification_the_office_dataset.tfrecord')
print(dataset)