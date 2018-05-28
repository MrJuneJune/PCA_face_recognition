# PCA_face_recognition

Using principle component analysis to recognize different faces.<\br>

Idea is that interested images can be recreated from a set of eigenfaces(eigenvectors of the training data) with weigth and mean image value. This is very close to fourier series but instead of sine and cosine, we are using eigenfaces and mean image.

--2018/05/28--
So far I calculated eigen faces and weights.
I need to find a clever way to calculate threshold value to classify different image sets. I am thinking of using false acceptance rate (FAR) and false match rate (FMR).

