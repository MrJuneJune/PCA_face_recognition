import os
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image
#%%
class ImageClass:
    
    # First it is going to grab image full of folder.
    def __init__(self):
        # Grabbing folder name
        root = tk.Tk()
        root.withdraw()
        self.folder_name = filedialog.askdirectory() 
        
    # Saving vectors in each row instead of column.
    def into_matrix(self, size, rows, array, folder_name):
        
        #Creating empty array to save vectorize images into.
        i_vectors = []
        
        # Vectorizing all images files
        for files_name in array:
            # reading images and coverting images into gray scaled value.
            img_plt = Image.open(folder_name+"/"+str(files_name)).convert('L')
            img = np.array(img_plt, 'uint8')
            if img is not None:
                if img.shape == 3:
                    img = cv2.cvtColor(
                            img,
                            cv2.COLOR_BGR2GRAY
                            )
                # import pdb; pdb.set_trace()
                i_vectors.append(np.ravel(img))

        return i_vectors 
            
    def vectorize(self):
        
        # find image sizes
        test_file = os.listdir(self.folder_name)[0]
        test_image = Image.open(self.folder_name+"/"+test_file).convert('L')
       	test_image = np.array(test_image, 'uint8')
           
        # Images in the folder
        images = [file_name for file_name in os.listdir(self.folder_name)]
        # Channels will be RGB value.
        height, width = test_image.shape
        size = height * width
    
        # Creating matrix to save vectorized images
        i_vectors = self.into_matrix(size, len(images), images, self.folder_name)
        
        return np.matrix(i_vectors), height, width
    
        
