import os
import numpy as np
import cv2
import random
from operator import itemgetter

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

def get_square(image, box):
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

folder = '/data/jhtlab/rzhang73/KITTI/vkitti_mask_data/train/car'
images = '/data/jhtlab/rzhang73/KITTI/vkitti_1.3.1_rgb/'
poses_dict = {}
num = 0

numCarsSeen = 0
for file in os.listdir(folder):
    # print(file)``
    if num % 500 == 0:
        print(num)
    num += 1
    anns = np.load(folder + '/' + file, allow_pickle = True, fix_imports = True).item()
    if 'poses' not in anns:
        print('bad file:', file)
        continue
    variation = anns['variation']

    poses = anns['poses']
    boxes = anns['bbxes']
    
    for i in range(len(poses)):
        pose = poses[i]
        box = boxes[i]
        filename = os.path.join(
            images, anns['world'], anns['variation'],
            '{:05d}'.format(anns['frame'])) + '.png'

        im = cv2.imread(filename, cv2.IMREAD_COLOR)

        im = im[:, :, ::-1]

        
        x1, x2, y1, y2 = get_crop_coordinates(im, box)

        #USE THIS FOR GETTING SQUARE AROUND CAR RATHER THAN TIGHT CROP
        # x1, x2, y1, y2 = get_square(im, box)

        cropped_im = im[y1:y2, x1:x2, :]

        #RESHAPES ALL IMAGES TO 108x108
        cropped_im = cv2.resize(cropped_im, (108, 108))


        pose = round(pose, 1)
        counter = 1
        if pose in poses_dict:
            counter = poses_dict[pose] + 1
            poses_dict[pose] += 1
        else:
            poses_dict[pose] = 1

        pose = round(pose, 3)
        ke = (pose, variation)
        counter = 1
        if ke in poses_dict:
            counter = poses_dict[ke] + 1
            poses_dict[ke] += 1
        else:
            poses_dict[ke] = 1
        pose = str(pose)


        cv2.imwrite('./data/vKitti_crop/' + pose + '#' + str(counter) + '_' + str(numCarsSeen) + '.jpg', cropped_im)
        # cv2.imwrite('./data/vKitti_no_crop/' + pose + '#' + str(counter) + '_' + str(numCarsSeen) + '.jpg', cropped_im)

        numCarsSeen += 1


