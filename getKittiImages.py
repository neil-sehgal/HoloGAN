import os
import numpy as np
import cv2
import random
from operator import itemgetter
import pickle

def get_crop_coordinates(image, box):
    """Gets tight crop with buffer of 5 px"""
    x1 = box[0]-5
    x2 = box[2]+5
    y1 = box[1]-5
    y2 = box[3]+5

    shape = np.shape(image)

    # Address out of bounds issues
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if x2 >= shape[1]:
        x1 -= x2 - shape[1]
        x2 = shape[1]
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if y2 >= shape[0]:
        y1 -= y2 - shape[0]
        y2 = shape[0]

    return x1, x2, y1, y2

def get_loose_crop(image, box):
    """Makes square around car"""
    x1 = box[0]
    x2 = box[2]
    y1 = box[1]
    y2 = box[3]

    width = x2-x1
    height = y2-y1
    buf_y = int((width-height)/2)
    y1 = y1-buf_y
    y2 = y2 + buf_y

    shape = np.shape(image)

    # Address out of bounds issues
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if x2 >= shape[1]:
        x1 -= x2 - shape[1]
        x2 = shape[1]
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if y2 >= shape[0]:
        y1 -= y2 - shape[0]
        y2 = shape[0]

    return x1, x2, y1, y2

folder = '/data/jhtlab/CSCI2951I/KITTI/training/processed'
images = '/data/jhtlab/CSCI2951I/KITTI/training/image_2'
poses_dict = {}
num = 0

numCarsSeen = 0
for file in os.listdir(folder):
    if num % 500 == 0:
        print(num)
    num += 1

    mf = folder + '/' + file
    with open(mf, 'rb') as pickle_file:
        anns = pickle.load(pickle_file)
    
    if len(anns) == 0:
        continue

    anns = [i for i in anns if i['type'] not in ['Truck', 'Van']]
    for theim_id in range(len(anns)):
        boxj = [int(x) for x in anns[theim_id]['bbox']]
        maskj = np.array([anns[theim_id]['mask']])
        maskj = np.array(np.reshape(maskj, maskj.shape[1:]),
                            dtype=np.int32)
        rotation = np.reshape((0, 0, 0), [1, 1, 3])
        distance = np.reshape(anns[theim_id]['location'], [1, 1, 3])
        pose = np.reshape(anns[theim_id]['pos'], (1, 1, 1))

        boxj = [int(x) for x in anns[theim_id]['bbox']]
        maskj = np.array([anns[theim_id]['mask']])
        maskj = np.array(np.reshape(maskj, maskj.shape[1:]),
                            dtype=np.int32)
        rotation = np.reshape((0, 0, 0), [1, 1, 3])
        distance = np.reshape(anns[theim_id]['location'], [1, 1, 3])
        pose = np.reshape(anns[theim_id]['pos'], (1, 1, 1))

        # Load file image data
        filename = os.path.join(images, mf.split('/')[-1].split('.')[0] + '.png')
        # print(filename)
        im = cv2.imread(filename, cv2.IMREAD_COLOR)
        assert im is not None, filename
        im = im[:, :, ::-1]


        x1, x2, y1, y2 = get_crop_coordinates(im, boxj)
        cropped_im = im[y1:y2, x1:x2, :]

        #RESHAPES ALL IMAGES TO 108x108
        resized_cropped_im = cv2.resize(cropped_im, (108, 108))


        pose = round(pose[0][0][0], 1)        
        pose = str(pose)


        # cv2.imwrite('./data/kitti_tight_crop/' + pose + '#' + str(numCarsSeen) + '_full.jpg', im)
        # cv2.imwrite('./data/kitti_tight_crop/' + pose + '#' + str(numCarsSeen) + '_resized.jpg', resized_cropped_im)
        cv2.imwrite('./data/kitti_tight_crop/' + pose + '#' + str(numCarsSeen) + '.jpg', resized_cropped_im)
        # cv2.imwrite('./data/kitti_tight_crop/' + pose + '#' + str(numCarsSeen) + '_crop.jpg', cropped_im)
        

        #Do it again for loose crop instead
        x1, x2, y1, y2 = get_loose_crop(im, boxj)
        cropped_im = im[y1:y2, x1:x2, :]

        #RESHAPES ALL IMAGES TO 108x108
        resized_cropped_im = cv2.resize(cropped_im, (108, 108))
        # cv2.imwrite('./data/kitti_loose_crop/' + pose + '#' + str(numCarsSeen) + '_resized.jpg', resized_cropped_im)
        cv2.imwrite('./data/kitti_loose_crop/' + pose + '#' + str(numCarsSeen) + '.jpg', resized_cropped_im)
        # cv2.imwrite('./data/kitti_loose_crop/' + pose + '#' + str(numCarsSeen) + '_crop.jpg', cropped_im)
        
        numCarsSeen += 1

   
