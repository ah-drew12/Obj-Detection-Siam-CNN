import tensorflow as tf
from tensorflow import io
import pandas as pd
import numpy as np
import os
from os import path as osp
import xml.etree.ElementTree as ET
import cv2

folder_path="samples"

pictures_xml=[folder_path+"/"+f for f in os.listdir(folder_path) if f[-1]=='l' and osp.isfile(osp.join(folder_path,f))]
pictures_jpg=[folder_path+"/"+f for f in os.listdir(folder_path) if f[-3:]=='jpg' and osp.isfile(osp.join(folder_path,f))]


writer = tf.io.TFRecordWriter('256_bounding_box_dataset_for_localization_13_frames.tfrecord')


tree=ET.parse("samples/test_1.xml")
root=tree.getroot()
tags = [elem.tag for elem in tree.iter()]

w=root[4][0].text#width
h=root[4][1].text#height
channels=root[4][2].text#rgb channel

quant_of_frames=len(root)-6


def load_img(img):
    img=tf.io.read_file(img)
    img=tf.image.decode_jpeg(img,channels=3)
    img=tf.cast(img,tf.float32)/255
    img=tf.image.resize(img,(256,256))
    return img

#coords for all test images
cords=[]
for xml,pic in zip(pictures_xml,pictures_jpg):
    #temp coords for particular image
    object_coords=np.zeros([1,4],dtype=np.float32)
    tree = ET.parse(xml)
    root=tree.getroot()
    for i in range(quant_of_frames):
        #xmin, ymin, xmax, ymax  у вигляді -1 до 1

        object=[]
        object.append(((float(root[6+i][4][0].text))/1310))
        object.append((float(root[6+i][4][1].text)/736))
        object.append((float(root[6+i][4][2].text)/1310))
        object.append((float(root[6+i][4][3].text)/736))
        if i==0:
            object_coords=np.reshape(np.array(object,dtype=np.float32),[1,4])
        else:
            object_coords=np.append(object_coords,np.reshape(np.array(object,dtype=np.float32),[1,4]),axis=0)



    img = load_img(pic)
    serialized_img = tf.io.serialize_tensor(img).numpy()
    serialized_cords = tf.io.serialize_tensor(object_coords).numpy()



    example = tf.train.Example(features=tf.train.Features(feature={
        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_img])),
        'cords': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_cords]))
    }))


    writer.write(example.SerializeToString())

writer.close()


dataset=tf.data.TFRecordDataset('256_bounding_box_dataset_for_localization_13_frames.tfrecord')
print(folder_path)