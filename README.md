Task:
The task is to implement a Siamese Neural Network (SNN) and train it with the given dataset for face recognition. You can use any face detection solution to obtain face crops and choose any backbone architecture for your SNN.

After training the model, perform face recognition on the given image with predefined instances. The final output image must include:
bounding boxes of all the detected faces;
labels to bounding boxes: character names for recognized instances and “undefined” for unrecognized ones.


Object Detection model were built with Siamese Convolution Neural Network and CNN localizator.
Localization of objects on image is doing on custom CNN Localization model with one photo that augmented into 10 different images.
When it has been trained, we started to train the classification model. The task requires a Siamese Neural Network that compares two images that are processed inside with the same 2 branches. In the end, the result of those branches concatenated and connected to a dense layer to output 
whether those 2 images are similar or not. The task requires to label the localizated frames, so, i decided to train a simple CNN classificator with characters from "The Office" because of Siamese NN cannot label the image with the needed class. Then i added that classification model to 
trained Siamese NN where the first image is going to simple classification model also it goes to one of two inputs Siamese NN. Result of the work of those models we get two outputs: classificator result and similarity of two images.
If classification model recognizes a character from "The Office" and Siamese NN finds similar a pair of the frame and verification image, then we label the frame with label of verification image. 

Dataset and model in .h5 format link: https://www.mediafire.com/file/ytsoy70plhv2jo1/git_hub_files.rar/file
