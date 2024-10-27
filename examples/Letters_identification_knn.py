import cv2
import os
import string
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
images = []
labels = []

def getLetterMap():
    letterMap={}
    for index,letter in enumerate(string.ascii_lowercase):
        idx = str(index)
        if index<10 :
            idx = "0"+str(index)
        letterMap[idx]  = letter
    return letterMap

def loadImages(data_directory, images, labels):
    letterMap = getLetterMap();
    for filename in os.listdir(data_directory):
        if filename.endswith(".png"):  # Adjust file extension if needed
            image_path = os.path.join(data_directory, filename)
            image = cv2.imread(image_path)
            letter =(filename.split("_"))[0]
            if(int(letter) <= 25):
                labels.append(letterMap[letter])
                image = cv2.resize(image, (224, 224))
                images.append(image)
                #images.append(img_resized)
image_dir ="/home/chandra/mldata/letters/letters"
loadImages(image_dir, images, labels)
images = np.array(images)
images = images.reshape(images.shape[0], -1)
model=KNeighborsClassifier(n_neighbors=3)
model.fit(images, labels)
image_path = os.path.join(image_dir, "02_31.png")  # Replace with your image path
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))
image = image.reshape(1, -1)
y_pred = model.predict(image)
print(y_pred)