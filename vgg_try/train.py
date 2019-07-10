import keras
import numpy as np
from keras.layers import Input,Convolution2D,MaxPooling2D,UpSampling2D
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation,Input,Flatten,Dense
from keras import backend as K
import io
import cv2
import os
import glob
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.models import Model
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint  
from keras.preprocessing.image import img_to_array  
import xml.dom.minidom 

K.set_image_dim_ordering("tf")

label_path = "E:/xiaoqi/ApplePests62/Annotations/"
img_path = "E:/xiaoqi/ApplePests62/JPEGImages/"
model_path = "E:/xiaoqi/models/"

labels = {}
label_cast = {"Alternaria_Boltch":0, "Brown_Spot":1, "Mosaic":2, "Grey_Spot":3, "Rust":4}

imgsize=512


for parent,dirnames,filenames in os.walk(label_path):
    for file in filenames:
        dom_ = xml.dom.minidom.parse(label_path+file)
        root = dom_.documentElement
        
        xmin = int(root.getElementsByTagName("xmin")[0].firstChild.data)
        ymin = int(root.getElementsByTagName("ymin")[0].firstChild.data)
        xmax = int(root.getElementsByTagName("xmax")[0].firstChild.data)
        ymax = int(root.getElementsByTagName("ymax")[0].firstChild.data)
        num = label_cast[root.getElementsByTagName("name")[1].firstChild.data]

        x = xmin
        y = ymin
        x_len = xmax-xmin
        y_len = ymax-ymin

        class_num = [0,0,0,0,0]
        class_num[num] = 1

        labels[file[:-4]] = [x,y,x_len,y_len] + class_num
print(labels)

def readimg(path,size=(512,512,3)):
    imglist = []
    Y = []
    for parent,dirnames,filenames in os.walk(path):
        for file in filenames:
            img_path = path+file
            imglist.append(img_path)
            Y.append(labels[file[:-4]])
            #img = cv2.imread(img_path)
            #img = cv2.resize(img,(512,512))
            #imglist[i] = img
    #imglist = np.reshape(imglist,[imglist.shape[0],512*512*3])
    train_X,test_X,train_Y,test_Y = train_test_split(imglist,Y,test_size = 0.2,random_state = 0)
    return train_X,test_X,train_Y,test_Y

def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = cv2.resize(img,(imgsize,imgsize))
        img = np.array(img,dtype="float") / 255.0
    return img

# data for training  
def generateData(batch_size,data=[]):  
    #print 'generateData...'
    leng = len("E:/xiaoqi/ApplePests62/JPEGImages/")
    while True:  
        train_data = []  
        train_label = []  
        batch = 0 
        for i in (range(len(data))): 
            url = data[i]
            batch += 1 
            img = load_img(url)
            img = img_to_array(img)
            train_data.append(img)  

            label = labels[url[leng:-4]]

            #label = load_img(filepath + 'label/' + url, grayscale=True)
            #label = img_to_array(label).reshape((img_w * img_h,))  
            # print label.shape  
            train_label.append(label)  
            if batch % batch_size==0: 
                #print 'get enough bacth!\n'
                train_data = np.array(train_data)  
                train_label = np.array(train_label)
                #train_label = labelencoder.transform(train_label)  
                #train_label = to_categorical(train_label, num_classes=n_label)  
                #train_label = train_label.reshape((batch_size,img_w * img_h,n_label))  
                yield (train_data,train_label)  
                train_data = []  
                train_label = []  
                batch = 0  

# data for validation 
def generateValidData(batch_size,data=[]):  
    #print 'generateData...'
    leng = len("E:/xiaoqi/ApplePests62/JPEGImages/")
    while True:  
        train_data = []  
        train_label = []  
        batch = 0  
        for i in (range(len(data))): 
            url = data[i]
            batch += 1 
            img = load_img(url)
            img = img_to_array(img)
            train_data.append(img)  

            label = labels[url[leng:-4]]

            #label = load_img(filepath + 'label/' + url, grayscale=True)
            #label = img_to_array(label).reshape((img_w * img_h,))  
            # print label.shape  
            train_label.append(label)  
            if batch % batch_size==0: 
                #print 'get enough bacth!\n'
                train_data = np.array(train_data)  
                train_label = np.array(train_label)
                #train_label = labelencoder.transform(train_label)  
                #train_label = to_categorical(train_label, num_classes=n_label)  
                #train_label = train_label.reshape((batch_size,img_w * img_h,n_label))  
                yield (train_data,train_label)  
                train_data = []  
                train_label = []  
                batch = 0  

#model
inputs = Input((imgsize, imgsize, 3))
conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)
conv5 = MaxPooling2D(pool_size=(2, 2))(conv5)
conv5 = Reshape([131072])(conv5)

den1 = Dense(500)(conv5)
den2 = Dense(100)(den1)

kua = Dense(4)(den2)

cla = Dense(5)(den2)
cla = Activation("softmax")(cla)

res = concatenate([kua, cla],axis=1)
model = Model(inputs=inputs, outputs=res)

model.summary()
#optimizer = SGD(lr=0.03, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer="adam", metrics=['accuracy'])

x_train,  x_test ,y_train, y_test= readimg(img_path)

EPOCHS = 20
BS = 8
modelcheck = ModelCheckpoint(model_path+"model1.h5",monitor='val_loss',save_best_only=True,mode='min')  
callable = [modelcheck]  
train_numb = len(x_train)  
valid_numb = len(x_test)  
print ("the number of train data is",train_numb)  
print ("the number of val data is",valid_numb)
model.fit_generator(generator=generateData(BS,x_train),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,  
                    validation_data=generateValidData(BS,x_test),validation_steps=valid_numb//BS,callbacks=callable,max_q_size=1)  
print("over!!!")
#model.save(model_path+"model1.h5")




