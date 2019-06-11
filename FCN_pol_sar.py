# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:46:31 2018

@author: Winggy Wu
"""
import argparse
import sys
import numpy as np  
from keras.models import Sequential  
from keras.layers import Reshape,Concatenate,Maximum,Add,Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation,Input  
from keras.utils.np_utils import to_categorical  
from keras.preprocessing.image import img_to_array  
from keras.callbacks import ModelCheckpoint  
from keras import callbacks
from sklearn.preprocessing import LabelEncoder  
from PIL import Image  
import matplotlib.pyplot as plt  
import cv2
import random
from keras.models import Model
import os
from tqdm import tqdm  
from keras import backend as k
#import keras.models as models
from keras.layers.core import Layer,Lambda
from keras.applications.resnet50 import ResNet50
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, SeparableConv2D
from skimage import io,color
from keras.models import load_model
from keras import optimizers
from sklearn.metrics import confusion_matrix
import itertools
from scipy import misc
import tensorflow as tf
import keras


Path=os.getcwd()
print(Path)
sar_data_path=Path+'/dataset/polsar_aug/' 

img_w = 256  
img_h = 256

n_label = 4+1
classes = [0. ,  1.,  2.,   3.  , 4.]
labelencoder = LabelEncoder()  
labelencoder.fit(classes)     
weightpath=Path+'/weights/'

def load_img(path, grayscale=False):
    if grayscale:
        
        img = io.imread(path)
        img = color.rgb2gray(img)
    else:
        img = io.imread(path)
        img = np.array(img,dtype="float") / 255.0
    return img
 

def get_train_val(path,val_rate = 0.25):
    train_url = []    
    train_set = []
    val_set  = []
    for pic in os.listdir(path + 'src'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i]) 
        else:
            train_set.append(train_url[i])
    return train_set,val_set


def generateData(path,batch_size,data=[]):  
    #print 'generateData...'
    while True:  
        train_data_pol = []
        train_data_sirv = []  
        train_label = []  
        batch = 0  
        for i in (range(len(data))): 
            url = data[i]
            batch += 1 
            img = load_img(path + 'src/' + url)
            img = img_to_array(img) 
            train_data_pol.append(img)
            
            texture = load_img(path + 'yam/' + url)
            texture = img_to_array(texture)  
            train_data_sirv.append(texture)
            
            label = load_img(path + 'label/' + url, grayscale=True) 
            label = img_to_array(label).reshape((img_w,img_h,))  
            
            train_label.append(label)  
            
            if batch % batch_size==0: 
               
                train_data_pol = np.array(train_data_pol)  
                train_data_sirv = np.array(train_data_sirv)
                
                train_label = np.array(train_label).flatten()  
                train_label = labelencoder.transform(train_label)   
                train_label = to_categorical(train_label, num_classes=n_label)  
                train_label = train_label.reshape((batch_size,img_w,img_h,n_label))  
                yield ([train_data_pol,train_data_sirv],train_label)  
                train_data_pol = []
                train_data_sirv=[]  
                train_label = []  
                batch = 0  



input_pol=Input((img_w, img_h,3), name="input")     
input_yam=Input((img_w, img_h,3), name="yam_input") 

base_model_pol = ResNet50(include_top=False, weights='imagenet', input_tensor=input_pol)
base_model_yam= ResNet50(include_top=False, weights='imagenet', input_tensor=input_yam)
base_model_pol.load_weights(weightpath+'usable/pretrain_weights3.hdf5',by_name=True)
base_model_yam.load_weights(weightpath+'usable/Yam.hdf5',by_name=True)
for layer in base_model_yam.layers:
    layer.name = layer.name + str("_yam")

def resize_bilinear(images):
    return k.tf.image.resize_bilinear(images, [img_h, img_w])
#polsar_branch
x2 = base_model_pol.get_layer('bn2c_branch2c').output
x3 = base_model_pol.get_layer('bn3d_branch2c').output
x4 = base_model_pol.get_layer('bn4f_branch2c').output
x5 = base_model_pol.get_layer('bn5c_branch2c').output

#yamaguchi_branch
s2 = base_model_yam.get_layer('bn2c_branch2c_yam').output
s3 = base_model_yam.get_layer('bn3d_branch2c_yam').output
s4 = base_model_yam.get_layer('bn4f_branch2c_yam').output
s5 = base_model_yam.get_layer('bn5c_branch2c_yam').output
#sum polsar and yam outputs
a2=Add()([x2, s2])
a3=Add()([x3, s3])
a4=Add()([x4, s4])
a5=Add()([x5, s5])

# Compress each skip connection so it has nb_labels channels.
c2=Convolution2D(n_label, (1, 1))(a2)
c3 = Convolution2D(n_label, (1, 1))(a3)
c4 = Convolution2D(n_label, (1, 1))(a4)
c5 = Convolution2D(n_label, (1, 1))(a5)

#return keras.backend.resize_images(images, img_h, img_w, "channels_last")
r2 = Lambda(resize_bilinear)(c2)
r3 = Lambda(resize_bilinear)(c3)
r4 = Lambda(resize_bilinear)(c4)
r5 = Lambda(resize_bilinear)(c5)

# Merge the three layers together using summation.
m = Add()([r2, r3, r4, r5])
##finish part
# Add softmax layer to get probabilities as output. We need to reshape
# and then un-reshape because Keras expects input to softmax to
# be 2D.
x = Reshape((img_h * img_w, n_label))(m)
x = Activation('softmax')(x)
x = Reshape((img_h, img_w, n_label))(x)
sar_model = Model(input=[input_pol,input_yam], output=x)
#sar_model.load_weights(weightpath+'usable/pretrain_weights3.hdf5', by_name=True)  
#        #conv10 = Conv2D(n_label, (1, 1), activation="sigmoid",name="conv10")(conv9)             #256*256*n_label
#sar_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #model.summary()    

train_set,val_set = get_train_val(path=sar_data_path,val_rate=0.2)
train_numb = len(train_set)
valid_numb = len(val_set)
EPOCHS = 200
BS = 8 #block_size 
weight_path=Path+'/weights/'+"3-Best-weights-pol-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"  
modelcheck = ModelCheckpoint(weight_path,monitor='val_acc',save_best_only=True,save_weights_only=True,mode='max')  
callable = [modelcheck]  
sar_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=1e-5,beta_1=0.99,beta_2=0.9999),
              metrics=['accuracy'])
H2 = sar_model.fit_generator(generator=generateData(sar_data_path,BS,train_set),
                        steps_per_epoch=train_numb//BS,
                        epochs=EPOCHS,
                        verbose=1,
                        validation_data=generateData(sar_data_path,BS,val_set),
                        validation_steps=valid_numb//BS,
                        callbacks=callable,
                        max_q_size=1)

#plot the training loss and accuracy
history_train_SAR=H2.history


plt.style.use("ggplot")
plt.figure(figsize=(7,5),dpi=900)
N = EPOCHS
plt.plot(np.arange(0, N), history_train_SAR["loss"], label="train_loss")
plt.plot(np.arange(0, N), history_train_SAR["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history_train_SAR["acc"], label="train_acc")
plt.plot(np.arange(0, N), history_train_SAR["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on FCN ")
plt.xlabel("Epoch #") 
plt.ylabel("Loss/Accuracy")
#plt.legend(loc="lower left")
plt.legend(loc="upper right") 
plt.savefig("plot_train_SAR_FCN")
file1 = Path +'trainfan.txt'
with open(file1,'w',encoding = 'utf-8') as wf:
    wf.write("loss: "+str(history_train_SAR["loss"])+'\n')
    wf.write("val_loss: "+str(history_train_SAR["val_loss"])+'\n')
    wf.write("acc: "+ str(history_train_SAR["acc"])+'\n')
    wf.write("val_loss: "+str(history_train_SAR["val_acc"])+'\n')


#history_train_SAR=H2.history
import csv
csvFile = open(Path+'/train_SAR_fcn.csv','w', newline='') 
writer = csv.writer(csvFile)
for key in history_train_SAR:
    writer.writerow([key, history_train_SAR[key]])
csvFile.close()

#%%评价指标确定 eval_segm:predict  gt_segm:label
def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i  = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i  += np.sum(curr_gt_mask)
 
    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_

def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
 
        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_

def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl   = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)
 
    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_
'''
Auxiliary functions used during evaluation.
'''
def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask

def extract_classes(segm): 
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl

def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")
'''
Exceptions
'''
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
#%% evaluate vision


img_w = 224  
img_h = 224
model_eva=sar_model
model_eva.load_weights(weightpath+'usable/3-Best-weights-pol-002-0.1754-0.9387.hdf5')#good
#model_eva.load_weights(weightpath+'3-Best-weights-pol-115-0.1255-0.9513.hdf5') #poor
#model_eva.load_weights(weightpath+'3-Best-weights-pol-032-0.2120-0.9313.hdf5')  #perfect
#model_eva.load_weights(weightpath+'1best_3-Best-weights-pol-195-0.0989-0.9607.hdf5') #perfect+
#model_eva.load_weights(weightpath+'3-Best-weights-pol-170-0.1087-0.9574.hdf5')
#model_eva.load_weights('I:/lili/Satellite-Segmentation-master/segnet/wupolsar_pack_tohailei/polsar_pack_tohailei/weights/3-Best-weights-pol-113-0.1249-0.9516.hdf5')

#image = load_img(sar_data_path+ 'src/' +'31.png')   
#image_yam = load_img(sar_data_path+ 'yam/' +'31.png') 
#image = load_img(sar_data_path+ 'src/' +'41.png')  
#image_yam = load_img(sar_data_path+ 'yam/' +'41.png')
 
image = load_img('I:/ieee-p/select/src/1.png')
image_yam = load_img('I:/ieee-p/select/yam/1.png') 
#image = np.resize(image,(256,256,3))
#image_yam = np.resize(image_yam,(256,256,3))


#image = load_img(aug_newsar_data_path+ 'false_polsar3.bmp') 
image = img_to_array(image)
image = np.expand_dims(image, axis=0) #(1,1024,1024,3)

image_yam = img_to_array(image_yam)
image_yam = np.expand_dims(image_yam, axis=0) #(1,1024,1024,3)

pred = model_eva.predict([image,image_yam],verbose=2)  #(1,1048576,5)        
pred1 = pred.reshape((img_h,img_w,n_label))#.astype(np.uint8)  #(1024,1024,5)
pred2_small = np.argmax(pred1,axis=2) #(1024,1024)
plt.xticks([])
plt.yticks([])
#plt.imshow(pred2)
#cv2.imwrite(Path+'/evaluation/u-net_6.png',pred2_small)#保存预测结果
cv2.imwrite('I:/ieee-p/select/pred1.png',pred2_small)
'''
label_image = load_img(sar_data_path+ 'label/' +'31.png', grayscale=True)
plt.xticks([])
plt.yticks([])
#plt.imshow(label_image)
'''
#%%

sum_pixelaccuracy=0
sum_meanaccuracy=0
sum_mIU=0

for url in val_set:
    img = load_img(sar_data_path + 'src/' + url) 
    img = img_to_array(img)  
    image = np.expand_dims(img, axis=0) #(1,1024,1024,3)
    
    img2 = load_img(sar_data_path + 'yam/' + url)
    img2 = img_to_array(img) 
    image2 = np.expand_dims(img2, axis=0) #(1,1024,1024,3)
    
    pred = model_eva.predict([image,image2],verbose=2)  #(1,1048576,5)       
    pred1 = pred.reshape((img_h,img_w,n_label))#.astype(np.uint8)  #(1024,1024,5)
    pred2 = np.argmax(pred1,axis=2) #(1024,1024)     
    
    label = load_img(sar_data_path + 'label/' + url, grayscale=True)
    label = img_to_array(label).reshape((img_w,img_h,))
    label = np.array(label).flatten()  
    label = labelencoder.transform(label)   
    label = label.reshape((img_w,img_h))
    
    sum_pixelaccuracy=sum_pixelaccuracy+pixel_accuracy(pred2,label)
    sum_meanaccuracy=sum_meanaccuracy+mean_accuracy(pred2,label)
    sum_mIU=sum_mIU+mean_IU(pred2,label)

sar_pixelacc=sum_pixelaccuracy/valid_numb
sar_meanacc=sum_meanaccuracy/valid_numb
sar_mIU=sum_mIU/valid_numb 

big_model=sar_model
#big_model.load_weights(Path+'/weights/usable/'+'weights-fcn-mod64-sar-038-0.1835-0.9372.hdf5') 
big_model.load_weights(Path+'/weights/usable/'+'3-Best-weights-pol-032-0.2120-0.9313.hdf5') 
#model=load_model(model_path)
#weight_path=PATH+'/pretrain_weights3.hdf5'
#model.save_weights(weight_path)
#model = resnet50_model_pol(224, 224, channel*num_img, num_classes)
#model.load_weights(weight_path, by_name=True)
image = load_img(Path+'/dataset/large/'+'img2.png')  
image1 = load_img(Path+'/dataset/large/'+'img2.png') 
#image = load_img(aug_newsar_data_path+ 'false_polsar3.bmp') 
#!!!resize the picture
image = np.resize(image,(256,256,3))
image1 = np.resize(image1,(256,256,3))

image = img_to_array(image)
image1 = img_to_array(image1)
#print(image.shape())
image = np.expand_dims(image, axis=0) 
image1 = np.expand_dims(image1, axis=0) 
pred = big_model.predict([image,image1],verbose=2)    
pred1 = pred.reshape((img_h,img_w,n_label))
pred2_big = np.argmax(pred1,axis=2) 
pred3_big = np.resize(pred2_big,(1024,1024))
plt.xticks([])
plt.yticks([])
#plt.imshow(pred2)
#!!! 
cv2.imwrite(Path+'/evaluation/large_test1.png',pred2_big)
#%% confusion_matrix
#model=load_model('Best-weights-my_model-008-0.0312-0.9910.hdf5')
#filepath1 = './0719sar_data/' 
filepath1 = Path+'/dataset/polsar_aug/'
data_dir_list = ['water','road','construction','vegetation']
for n in range(50):
    image = load_img(filepath1 + 'src/' +'%d.png'%n) 
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) 
    pred = model_eva.predict([image,image],verbose=2)        
    pred1 = pred.reshape((img_h,img_w,n_label)).astype(np.uint8)  
    pred2 = np.max(pred1,axis=2)  
    pred3 = img_to_array(pred2)  
    pred = pred3.flatten()
        
    label = load_img(filepath1 + 'label/' +'%d.png'%n,grayscale=True) 
    label1 = img_to_array(label).reshape(img_w*img_h) 
    label2 = label1.astype(np.uint8)
    label = np.array(label2).flatten()
#print(confusion_matrix(np.argmax(Y_valid,axis=1), y_pred))
    confmat = confusion_matrix(y_true=label,y_pred=pred)
#    print(confmat)
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cnf_matrix = (confusion_matrix(y_true=label, y_pred=pred))

np.set_printoptions(precision=4)

fig=plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=data_dir_list,
                      title='Confusion matrix')
plt.show()
#fig.savefig('confusion_matrix.jpg')
fig.savefig(Path + '/evaluation/confusion_matrix.png')











