import cv2
import csv
import random
import keras
from keras.datasets import mnist
import numpy as np
from keras.layers import Input,Convolution2D,MaxPooling2D,UpSampling2D
from keras import Model
from keras import backend as K
from keras.models import load_model
import cv2
import os,sys
from sklearn.model_selection import train_test_split

img_path = ""
img_new = ""

label_cast = {"Alternaria_Boltch":0, "Brown_Spot":1, "Mosaic":2, "Grey_Spot":3, "Rust":4}

for i in range(6291):
    name,x,y,label = reader[i]
    print(name,label)
    if label in labels_1:
        img = cv2.imread(img_path+name+".jpg")
        if img is None:
            continue
        print(cv2.imwrite(img_new+"1/"+name+".jpg",img))
    elif label in labels_0:
        img = cv2.imread(img_path+name+".jpg")
        if img is None:
            continue
        print(cv2.imwrite(img_new+"0/"+name+".jpg",img))
    else:
        print("error!!!")
        exit(1)


filename = "E:/xiaoqi/vgg_try/label.txt"
f = open(filename)
label = f.readline()

K.set_image_dim_ordering("th")

img_path = "E:/xiaoqi/ApplePests62/test/"
save_path = "E:/xiaoqi/ApplePests62/predict/"

def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path,-1)
        img = np.reshape(img,[img.shape[0],img.shape[1],1])
        img = np.array(img,dtype="float") / 255.0
    return img


#x_train = np.reshape(x_train, (len(x_train), 512, 512, 3))
#x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))

#noise_factor = 0.5
#x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
#x_test_noisy = np.clip(x_test_noisy, 0., 1.)

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

model = load_model("E:/xiaoqi/models/model1.h5")

test_img = cv2.imread(img_path)
x_test = x_test.astype('float32') / 255.
result = model.predict(test_img)

x = [result[0],result[0]+result[1],result[2],result[2]+result[3]]
label_num = np.argmax(result[4:])
plot_one_box(x,img,color = None,label=label_cast[label_num])



for name,x,y,label in reader:
    print(name)
    img = cv2.imread(new+name+".jpg")
    x = int(x)
    y = int(y)
    x = [label[0],label[1],label[2],label[3]]
    label_num = np.argmax(label[4:])
    plot_one_box(x,img,color = None,label=label_cast[label_num])
    print(cv2.imwrite(cate+label+"/"+name+".jpg",img))


