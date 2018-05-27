# -*- coding: utf-8 -*-
"""
Created on Mon May 21 20:21:28 2018

@author: June

Name: PCA Face Recognition
"""
import cv2
import os
import numpy as np
from numpy import linalg as LA
#from sklearn.decomposition import PCA
#%%
cropping = False
refpt = []

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
     for i in len(v):
          distance = distance**2 + v[i]**2
     eu_distance = np.sqrt(distance)
     return eu_distance
#%%
# Capturing default web cam
seconds = 0
count = 0

if  os.path.isdir("My_Faces") == False:
     while(True):
          cap = cv2.VideoCapture(0)
          # Using try since easy to close if there is an error.
          # Reading default webcam
          ret, frame = cap.read()
          cv2.imshow("frame", frame)
          cv2.setMouseCallback("frame", cc)
          if len(refpt) == 2:
               # Region of interest cropped
               roi = frame[refpt[0][1]:refpt[1][1], refpt[0][0]:refpt[1][0]]
               cv2.imshow("roi",roi)
               success = True
               while success:
                    cv2.imwrite("face%d.jpg" % count, roi)     # save frame as JPEG file      
                    success,image = cap.read()
                    print('Read a new frame: ', ret)
                    count += 1
                    if count >= 10:
                         break
          key = cv2.waitKey(1) & 0xFF
          if key == ord("q"):
               break
          cap.release()
          cv2.destroyAllWindows()

#%%
i = 0
h = len([name for name in os.listdir('My_Faces')])
w1 = 2
w = 34560
i_vectors = [[0 for x in range(w)] for y in range(h)] 
num_faces = [[0 for x in range(w1)] for y in range(h)] 
for file_name in os.listdir("My_Faces"):
     img = cv2.imread("My_Faces/"+str(file_name))
     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
     num_faces[i] = img
     i_vectors[i] = np.ravel(img)
     i += 1
#i_vectors = np.transpose(i_vectors)
#%%
# Finding Mean Value
sums = np.zeros(w)
for i in range(h):
     sums = sums + i_vectors[i]
     print(sums)

# Need to be in unit8 value to be shown     
mean = np.uint8(sums/h)
#%%
# Showing Mean value 
# Changing to uint8 form to show average faces
A_face = mean.reshape(192,180)
cv2.imshow( "Average_Face", A_face );

# Change back to float
mean_float = np.float(mean)
#%%
     # Matrix with mean value substracted
B = [[0 for x in range(h)] for y in range(w)] 
for i in range(h):
     B[i] = i_vectors[i] - mean_float
     # Since negative value should be equal to zero in uint8
#     B[i] = B[i].clip(min=0)
#     B[i] = np.uint8(B[i])
     
#%%
 #Finding covariacne matrix
S = (1/(h-1))*np.dot(np.transpose(B),B)
S = np.uint8(S)
#%%
# Find eigenvector
#w, v = LA.eig(S)
#%%
 #Got stuck here...
#eigen_faces = 
#%%
#pca = PCA()
#pca.fit(i_vectors)
#e_vectors = pca.components_
##eigenvalues = pca.explained_variance_
## a = np.arange(6).reshape((3, 2))
##%%
#e_face = np.uint8(np.uint8(e_vectors[0].reshape(192,180)))
#cv2.imshow( "Eigen Face", e_face );
#key = cv2.waitKey(1) & 0xFF
cv2.waitKey(10000)
cv2.destroyAllWindows()

          

