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
     for i in range(len(v)):
          distance = distance**2 + v[i]**2
     eu_distance = np.sqrt(distance)
     return eu_distance
#%%
# Capturing default web cam
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
            success = True
            while success:
                cv2.imwrite(os.path.join(super_folder,"face%d.jpg") % count, roi)     # save frame as JPEG file      
                success,image = cap.read()
                print('Read a new frame: ', ret)
                count += 1
                if count == 50:
                    break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

#%%
# Show test files
test_file = os.listdir(super_folder)[0]
test_image = cv2.imread(super_folder+"/"+test_file)
cv2.imshow("Image", test_image)
#%%
i = 0
h = len([name for name in os.listdir(super_folder)])
height, width, channels = test_image.shape
size = height * width
i_vectors = [[0 for x in range(size)] for y in range(h)] 
num_faces = [[0 for x in range(2)] for y in range(h)] 
for file_name in os.listdir(super_folder):
     img = cv2.imread(super_folder+"/"+str(file_name))
     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
     num_faces[i] = img
     i_vectors[i] = np.ravel(img)
     i += 1
#i_vectors = np.transpose(i_vectors)
#%%

# Finding Mean Value
sums = np.zeros(size)
for i in range(h):
     sums = sums + i_vectors[i]

# Need to be in unit8 value to be shown     
mean = np.uint8(sums/h)

# float
mean_float = sums/h
#%%
# Showing Mean value 
# Changing to uint8 form to show average faces
A_face = mean.reshape(height, width)
cv2.imshow( "Average_Face", A_face );
#%%
# Matrix with mean value substracted
B = [[0 for x in range(size)] for y in range(h)] 
for i in range(h):
     B[i] = i_vectors[i] - mean_float
B_array = np.array(B)     
#%%
 #Finding covariacne matrix
# B.B^T  instead of B^T.B since it will take too long to calculate and you only need biggest eigen vector
# of B^T.B and it can be done by using eigen vector of B.B^T 
S = (1/(h-1))*np.dot(B_array,np.transpose(B_array))

# Find eigenvector
w, v = LA.eig(S)
#%%
# Calculating eigen_faces
eigen_faces = np.dot(np.transpose(B_array), v)
eigen_faces = eigen_faces.clip(min=0)
eigen_faces = np.uint8(eigen_faces)

#Normalizing
#eigen_faces_float = np.float64(eigen_faces)
## To show images(uint8 data)
#eigen_faces_uint8 = eigen_faces.clip(min=0)
#eigen_faces_uint8 = np.uint8(eigen_faces_uint8)
##%%
#eigen_faces_eu = eucliean(eigen_faces_float[:,0])
#eigen_face_norm = eigen_faces_float[:,0] / eigen_faces_eu(axis=0)
#%%
# Showing Eigenfaces
cv2.imshow("faces", eigen_faces[:,0].reshape(height,width))
#%%
i=0
#  Test image sets using to be classified(Default Face)
test_img = [[0 for x in range(size)] for y in range(6)]
test_img_m = [[0 for x in range(size)] for y in range(6)]
for file_name in os.listdir(super_folder)[:6]:
     img = cv2.imread(super_folder+"/"+str(file_name))
     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
     test_img[i] = np.ravel(img)
     test_img_m[i] = i_vectors[i] - mean_float
     i += 1
# So that column is the vector image 
test_img_m = np.transpose(np.array(test_img_m))
#%%
# Classifying my face.
# Only going to use first 6 eigen faces to reconstruct my face.
top_six = np.zeros((size,6))
weight = np.zeros((6,1))
for i in range(6):
    weight[i] = np.dot(np.transpose(eigen_faces[:,i]), test_img_m[:,i])
#%%
# Probe Image to recognize
for i in range(25):
    # Trying to find threshold value for Image to recognize
    probe_file = os.listdir(super_folder)[i]
    probe_img = cv2.imread(super_folder+"/"+str(probe_file))
    probe_img = cv2.cvtColor(probe_img, cv2.COLOR_BGR2GRAY) 
    probe_img_m = np.ravel(probe_img) - mean_float
    # Probe weights
    probe_weight = np.zeros((6,1))
    for c in range(6):
        probe_weight[c] = np.dot(np.transpose(eigen_faces[:,c]), probe_img_m)
    # Comparing weights by using euclidean distance.
    # Could have used  mahalanobis distance which is better for pattern recognition.
    weight_c = np.zeros((6,1))
    for z in range(6):
        weight_c[z] = np.sqrt( (weight[z] - probe_weight[z])**2) 
    weight_c_min = np.min(weight_c)
    print(weight_c_min)
    if weight_c_min <= 3923792:
        print("Yes, Default Face")
    else:
        print("No, it is not Default Face")
cv2.waitKey(20000)
cv2.destroyAllWindows()