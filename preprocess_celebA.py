#Removes greyscale images from celebA dataset
import os
import numpy as np 
from PIL import Image

shapeDict = {}
idx = 0
for filename in os.listdir('./data/celebA'):
    idx += 1
    img = Image.open('./data/celebA/' + filename)
    data = np.asarray(img)
    shp = data.shape
    if(len(shp)!=3):
        print(shp)
        print(filename)
        os.remove('./data/celebA/' + filename)