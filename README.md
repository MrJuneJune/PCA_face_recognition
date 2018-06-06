# PCA_face_recognition

What is PCA? : Principle component analysis is a statistical procedure that uses an orthogonal transformation to convert a set of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.

Short definition: it looks for variables or traits that is more correlated than others to reduce dimensions of the data. This variable is called principal component also known as eigenvectors. 

For example, when we look at anything kind of sporting events, we don't usually consider z-axis of the events. It is because z-axis does not make what we are seeing more believable. In fact, z-axis can make things more hectic, causing motion sickness. This is also reason why 3-D movies are underwhelming to watch.

<img src="https://media.giphy.com/media/lZoPvIJZfEREI/giphy.gif" style="align:center;"/>
In basketball game, spectator can't really see how high the basketball travels.

PCA removes inefficient dimensions and creates more important variable(principal component). This is particularly useful to analyze data with lots of variables  to reduce computational efforts such as stock. For today, I will be using PCA  to create face recognition programs.

Requirement.

- Python 3.6.4
- Opencv 3.4.1
- numpy

To begin with, I need a folder of Imaging data to test and classify. To do this, I used my webcam to capture images and save them in a folder. 

while(True):
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    cv2.setMouseCallback("frame", cc)
    if len(refpt) == 2:
        # Region of interest cropped
        roi = frame[refpt[0][1]:refpt[1][1], refpt[0][0]:refpt[1][0]]
        cv2.imshow("roi",roi)

In the code, cc is a defined function that I created to crop the region of interested area. 

After I capture the images, I ravel the image into series of vectors and created a matrix from the vectors. If there is a <bold> X amounts of  M by N  size Images </bold>, it is stored in <bold> X by M*N Matrix (since it is saved as array in each row)</bold>. These image needs to be gray scaled.

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


From it, I need to find a mean value of the image set to center the data to origin.

-Finding Mean Value
sums = np.zeros(size)
for i in range(h):
     sums = sums + i_vectors[i]
-Need to be in unit8 value to be shown     
mean = np.uint8(sums/h)
-float
mean_float = sums/h
-Showing Mean value 
-Changing to uint8 form to show average faces
A_face = mean.reshape(height, width)
cv2.imshow( "Average_Face", A_face );


My average faces look something like this.

<img src="https://i.imgur.com/KqtSeVX.png"/>


Now that I have a mean value, I need to subtract mean value from the data to center the data around the origin <bold>(defined as B)</bold>. Then, covariance matrix of the data is computed. When I was computing this, I intentionally used <bold> B.B^T </bold> since there will be less values to compute <bold> (Keep in mind that B is X by M*N)</bold> so that covariance matrix will be in <bold> X by X </bold> matrix.

-Finding covariance matrix
S = (1/(h-1))*np.dot(B_array,np.transpose(B_array))


From the covariance matrix, I can compute eigen vectors.


-Find eigenvector a.k.a PC
w, v = LA.eig(S)


However, these eigenvectors will be eigenvectors of <bold> X by X </bold> matrix instead of <bold> M*N by M*N </bold> matrix. By using matrix properties, we can calculate eigenvector of <bold> M*N by M*N</bold>. Keep in mind that eigen_faces need to be in unit vector forms.


-Calculating eigen_faces
eigen_faces_float = np.dot(np.transpose(B_array), v)
x, y = eigen_faces_float.size
#%%
-Make it into unit vector
eigen_faces_norm = [[0 for x in range(x)] for y in range(6)] 
magnitude = np.zeros(x)
-10 best eigen vectors and check if the egien values are in unit vector value
for i in range(6):
    eigen_faces_norm[i] = eigen_faces_float[:,i]
    for j in range(x):
        magnitude[j] = eigen_faces_norm[i][j]^2
    magnitude_sum = np.sqrt(np.sum(magnitude))
    for j in range(x):
        eigen_faces_norm[i][j] = eigen_faces_norm[i][j]/magnitude_sum
    -To check if they are in unit vector form.
        magnitude[j] = eigen_faces_norm[i][j]^2
    magnitude_sum = np.sqrt(np.sum(magnitude))
    print(magnitude_sum)
-So I can transelate into unit8 data type
eigen_faces = eigen_faces_float.clip(min=0)
eigen_faces = np.uint8(eigen_faces)
#%%
-Showing Eigenfaces
cv2.imshow("faces", eigen_faces[:,0].reshape(height,width))


This is what I got for one of the eigenfaces.


<img src="https://i.imgur.com/Zj4I9Ol.png"/>


Now, I have to find weights for different training sets of images which is simple as multiplying transposed eigen faces with test images (with mean substracted). This is basically means that how much of these eigenvectors are need to create that test images. This weight is going to be threshold value to decide what the image is.
 

-Classifying my face.
-Only going to use first 6 eigen faces to reconstruct my face.
top_six = np.zeros((size,6))
weight = np.zeros((6,1))
for i in range(6):
    weight[i] = np.dot(eigen_faces_norm[i], test_img_m[:,i])
    magnitude[i] = weight[i]^2
-Threshold values
print(np.sqrt(np.sum(magnitude)))


I have gather all the information to test if it works. It had 87% chance of getting correct faces (23/25) which is resonable.

-Probe Image to recognize
for i in range(25):
    # Trying to find threshold value for Image to recognize
    probe_file = os.listdir(super_folder)[i]
    probe_img = cv2.imread(super_folder+"/"+str(probe_file))
    probe_img = cv2.cvtColor(probe_img, cv2.COLOR_BGR2GRAY) 
    probe_img_m = np.ravel(probe_img) - mean_float
    # Probe weights
    probe_weight = np.zeros((6,1))
    for c in range(6):
        probe_weight[c] = np.dot(eigen_faces_norm[c], probe_img_m)
    # Comparing weights by using euclidean distance.
    weight_c = np.zeros((6,1))
    for z in range(6):
        weight_c[z] = (weight[z] - probe_weight[z])^2
    weight_c_min = np.sqrt(np.sum(weight_c))
    print(weight_c_min)
    if weight_c_min <= np.sqrt(np.sum(magnitude)):
        print("Yes, Default Face")
    else:
        print("No, it is not Default Face")

Problem with PCA: PCA will try to reconstruct any data that is given if it was not represented in the data which will give bizarre reconstruction.

