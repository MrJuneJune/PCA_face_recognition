# -*- coding: utf-8 -*-
"""
Created on Mon May 21 20:21:28 2018

@author: June

Name: PCA Face Recognition
"""
import cv2
import os
import numpy as np
import vectorize as vc 
from numpy import linalg as LA
from matplotlib import pyplot as plt
#%%
# Cropping out Region of Area wanted
def cc(event, x, y, flags, param):
     global refpt, cropping
     if event == cv2.EVENT_LBUTTONDOWN:
          refpt = [(x, y)]
          cropping = True
     elif event == cv2.EVENT_LBUTTONUP:
          refpt.append((x,y))
          cropping = False
          
     return refpt
 
# Defining Eucliean Distance
def eucliean(v):
     distance = 0
     for i in range(len(v)):
          distance = distance**2 + v[i]**2
     eu_distance = np.sqrt(distance)
     return eu_distance
#%%
cropping = False
refpt = []
# Capturing default web cam
while True:
    answer = input('Do you want to take pictures with your webcam ? [y/n] ')
    if answer == 'y':
        count = 0
        super_folder = input("Name your supervise group: ")
        if  os.path.isdir(super_folder) == False:
            os.makedirs(super_folder)
            cap = cv2.VideoCapture(0)
            while(True):
                ret, frame = cap.read()
                cv2.imshow("frame", frame)
                cv2.setMouseCallback("frame", cc)
                if len(refpt) == 2:
                    # Region of interest cropped
                    roi = frame[refpt[0][1]:refpt[1][1], refpt[0][0]:refpt[1][0]]
                    cv2.imshow("roi",roi)
                    key = cv2.waitKey(1000) & 0xFF
                    success = True
                    while success:
                        cv2.imwrite(os.path.join(super_folder,"face%d.jpg") % count, roi)     # save frame as JPEG file      
                        success,image = cap.read()
                        print('Read a new frame: ', ret)
                        count += 1
                        if count == 50:
                            break
                    break
            cap.release()
            cv2.destroyAllWindows()
    elif answer == 'n':
        print('Select existing supervise folder')
        break
    else:
        print('That is not a valid answer...')
        continue

#%%
# Select a folder.
images = vc.ImageClass()
i_vectors, height, width = images.vectorize()
# Finding mean values. This values will be in float dtype instead of uint8 which it came from. 
# To change back to image, we need to convert it back to uint8 dtype.
mean = i_vectors.mean(0)
while True:
    answer = input('Do you want to see what your mean face look like? [y/n] ')
    if answer == 'y':
        plt.imshow(mean.reshape(height, width), cmap='gray')
        break
    elif answer == 'n':
        break
    else:
        print('Not a valid response')
        continue
#%%
# Matrix with mean value substracted
B = np.subtract(i_vectors, mean)
number_of_images, vector_length = B.shape
#%%
 #Finding covariacne matrix
# B.B^T  instead of B^T.B since it will take too long to calculate and you only need biggest eigen vector
# of B^T.B and it can be done by using eigen vector of B.B^T 
S = (1/(number_of_images-1))*np.dot(B,np.transpose(B))

# Find eigenvector
w, v = LA.eig(S)
#%%
# Calculating eigen_faces
eigen_faces_float = np.dot(np.transpose(B), v)
# So I can transelate into unit8 data type
eigen_faces = eigen_faces_float.clip(min=0)
eigen_faces = np.uint8(eigen_faces)
x, y = eigen_faces_float.shape
#%%
# Make it into unit vector
n_pca = int(input("How many number of PCA do you want to use ? "))
eigen_faces_norm = [[0 for x in range(x)] for y in range(n_pca)] 
magnitude = np.zeros(x)
magnitude = magnitude.astype(complex)
#%%
# 10 best eigen vectors and check if the egien values are in unit vector value

for i in range(n_pca):
    eigen_faces_norm[i] = eigen_faces_float[:,i]
    for j in range(x):
        magnitude[j] = eigen_faces_norm[i][j]**2
    magnitude_sum = np.sqrt(np.sum(magnitude))
    for j in range(x):
        eigen_faces_norm[i][j] = eigen_faces_norm[i][j]/magnitude_sum
    # To check if they are in unit vector form.
        magnitude[j] = eigen_faces_norm[i][j]**2
    magnitude_sum = np.sqrt(np.sum(magnitude))
    print(magnitude_sum)
#%%
# Test Images sets
print('Please Select test images folder!')
test_images = vc.ImageClass()
testi_vectors, th, tw = test_images.vectorize()
#%%
# Substract mean value from test image sets.
testi_vectors = np.subtract(testi_vectors, mean)
testi_vectors_m = np.transpose(testi_vectors)
#%%
# Classifying my face.
# Only going to use first 6 eigen faces to reconstruct my face.
weight = np.zeros((n_pca,1))
magnitude = np.zeros((n_pca,1))
for i in range(n_pca):
    weight[i] = np.dot(np.transpose(eigen_faces_norm[i]), testi_vectors_m[:,i])
    magnitude[i] = weight[i]**2
print(np.sqrt(np.sum(magnitude)))
#%%
# Reconstruction.
recontruction = [[0 for x in range(x)] for y in range(n_pca)] 
for i in range(n_pca):
    recontruction[i] = weight[i]*np.transpose(eigen_faces_norm[i])
    recontruction[i] = recontruction[i].clip(min=0)
    recontruction[i] = np.uint8(recontruction[i])
#%%
#Finding Probe Images.
print("Please Select Probe Images")
probe_images = vc.ImageClass()
probe_images_m, prh, prw = probe_images.vectorize()
probe_images_m = probe_images_m - mean
probe_weights = np.zeros((n_pca,1))
#%%
for i in range(len(probe_images_m)):
    for c in range(n_pca):
        probe_weights[c] = np.dot(np.transpose(eigen_faces_norm[c]), np.transpose(probe_images_m[i]))
    weight_c = np.zeros((n_pca,1))
    for z in range(n_pca):
        weight_c[z] = (weight[z] - probe_weights[z])**2
    weight_c_min = np.sqrt(np.sum(weight_c))
    plt.figure()
    plt.imshow(probe_images_m[i].reshape(height,width))
    print("Probe image ", i, "is" )
    if weight_c_min <= np.sqrt(np.sum(magnitude)):
        print("interested face")
    else:
        print("not interested face")
#%%
plt.imshow(testi_vectors.mean(0).reshape(height, width), cmap='gray')
