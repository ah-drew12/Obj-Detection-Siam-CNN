#
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import Sequential, Model, Input
import tensorflow.compat.v1 as teff
from keras.models import clone_model,load_model
import cv2
import matplotlib.pyplot as plt


#This script executes an object detection model, where object localization and object classification are connected.


teff.enable_eager_execution(teff.ConfigProto(log_device_placement=True))
gpus = tf.config.list_logical_devices('GPU')

tf.config.run_functions_eagerly(True)

dataset = tf.data.TFRecordDataset('classification_dataset_v1_inside_modified_siamese_cnn.tfrecord')
#REading images and labels from tfrecord files
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


#Loading pretrained models
localizator=load_model('D:/tmp_modified_classificatior/localizator_weights_25_01/1/222/22/223/test111_bounding_box_for_many_objects.h5')
classificator=load_model("modified_siamese_cnn_testttt.h5")

# image_for_test= cv2.imread('input_image.jpg')

# image_for_test=cv2.cvtColor(image_for_test, cv2.COLOR_BGR2RGB)

def load_img(img):
    img=tf.io.read_file(img)
    img=tf.image.decode_jpeg(img,channels=3)
    img=tf.cast(img,tf.float32)/255
    img=tf.image.resize(img,(256,256))
    return img
#Loading test images
image_for_test=load_img('input_image.jpg')
# image_for_test=image_for_test.astype(dtype=np.float64)/256
#Output images for test
plt.figure(0)
plt.imshow(image_for_test)

#Getting location of frame boxes
pred=localizator.predict(tf.reshape(image_for_test,[-1,256,256,3]))
#Reshaping coordinates from one array to formax (n,4) where 4 is minX,minY,maxX,maxY
pred = np.reshape(pred, (int(pred.size / 4), 4))
#paint  and crop frames
def crop_frames(pred,image):
    #Function for cropping frames from image by coordinates that have been gotten from localization model
    new_image = np.zeros([256, 256, 3], dtype=np.float64)
    i = np.copy(image[:, :, :])
    image_for_frame=np.copy(image[:, :, :])
    frames_amount=int(pred.shape[0])

    pred = tf.reshape(pred, [-1, frames_amount, 4])
    c = pred[0]

    # ax=plt.subplot(1,1,num+1)
    i = ((i) * 255).astype(dtype=np.float64)

    c = (c.numpy()*255).astype(dtype=np.int16)

    # for predicted
    color = (255, 255, 0)
    # frames=np.zeros((frames_amount, 1 , 1, 1))

    frames=[]
    temp_frame=[]
    # for citeration in range(len(c)):
    for bb,iteration in zip(c,range(len(c))):
        temp_frame = image_for_frame[bb[1]:bb[3], bb[0]:bb[2]]
        # temp_frame[iteration]
        # plt.figure()
        #
        # plt.imshow(temp_frame)
        #
        # plt.show()

        frames.append(temp_frame)

        i=cv2.rectangle(i, (bb[0], bb[1]), (bb[2], bb[3]), color=(255, 255, 0), thickness=1)

    frames=np.array(frames)

    # plt.figure()
    final_image = (i) / 255
    final_image = tf.convert_to_tensor(final_image, dtype=tf.float64)
    plt.figure(1)
    plt.imshow(final_image)
    plt.show()

    print('image outputed')
    return frames

frames = crop_frames(pred,image_for_test)


def create_embedding(images,labels,frame):
    #Creating dataset for each frame. Mixing frame with verification images and setting a label for each pair.
    #images - set of images with labels
    #frame - needed image to classify
    x, y = [], []
    d = zip(x, y)
    tuples = [(x1, y1) for x1, y1 in zip(images, labels)]
    count = 0
    embedding = ['angela_martin','creed_bratton', 'dwight_shrute', 'jim_halpert', 'kelly_kapoor', 'kevin_malone',
                 'meredith_palmer' ,'michael_scott', 'oscar_martinez' ,'pam_beesly', 'stanley_hudson', 'unknown']
    for t in tuples:

        new_label = np.zeros(13)
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
        x.append([np.resize(frame,(256,256,3)),np.resize(t[0],(256,256,3))])
    x=np.array(x)
    y=np.array(y)
    embedding=np.array(embedding)
    return x,y,embedding



pred_embed=[]

for iter,frame in enumerate(frames):
    #Creating pairs with images for each frame
    x_pairs, y_labels, embeddings = create_embedding(images, labels, frame)
    # frame=frames[1]
    # Prediction a label for each frame
    temp_pred=classificator.predict([x_pairs[:,0],x_pairs[:,1]])
    # temp_pred=[temp_pred[:,i] for ]
    #Exporting result of classification for each pair
    pd.DataFrame(temp_pred).to_csv('prediction_For_frame22222222.csv')
    pred_embed.append(temp_pred)
    #Reducing array to delete label "unmatch"
    temp_pred=temp_pred[:,:12]
    #Evaluate average for each column. As we get about 130 pairs where about 10 images for each label,
    # we should have bigger probability for "unmatch", so we deleting it from analyse and search maximum among the others labels.
    #If label has probability higher than 0.1 then we setting this label as predicted
    averages=np.average(temp_pred,axis=0)
    GET_AVERAGES=[1 if ((average>0.1) & (average==np.max(averages))) else 0 for average in averages]
    pred_label=embeddings[np.argmax(GET_AVERAGES,axis=0)]

    #Adding text to test image
    plt.figure(1)
    plt.text(pred[iter,0]*255,pred[iter,1]*255-1 , pred_label, fontsize=10,color='r')
    plt.figure()
    plt.imshow(frame)
    plt.show()

pred_embed=np.array(pred_embed)

#each frames goes into the classification model and return softmax label of class
print('d')