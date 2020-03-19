import os
import numpy as np
import cv2
import random
from operator import itemgetter

def get_crop_coordinates(image, box):
    """Just gets tight crop with buffer of 5 px"""
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

folder = '/data/jhtlab/rzhang73/KITTI/vkitti_mask_data/train/car'
images = '/data/jhtlab/rzhang73/KITTI/vkitti_1.3.1_rgb/'
poses_dict = {}
# dirs = set()
num = 0
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
    # print(anns['world'])
    # print(anns['variation'])
    # print(anns['frame'])
    for i in range(len(poses)):
        pose = poses[i]
        box = boxes[i]
        filename = os.path.join(
            images, anns['world'], anns['variation'],
            '{:05d}'.format(anns['frame'])) + '.png'

        im = cv2.imread(filename, cv2.IMREAD_COLOR)

        im = im[:, :, ::-1]

        
        x1, x2, y1, y2 = get_crop_coordinates(im, box)
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


        cv2.imwrite('./data/vKitti/' + pose + '#' + str(counter) + '.jpg', cropped_im)




# def get_crop_coordinates_old(image, boxj, box_size=None,
#                          translate=True):
#     """
#     Returns the coordinates (x1, y1) and (x2, y2) that represents
#     where the images should be cropped to have a square image
#     with the object in question.

#     The coordinates form a square image, centered on the object.
#     Args:
#         image (2d array): image in question
#         boxj ([1, 4] array): bounding box of object
#         box_size (int): by default, the size of the crop box is
#             randomized for data augmentation. When box_size is
#             specified, the box size will be fixed.
#         translation (bool): by default, the box will be translated
#             by self.augment_translate for data augmentation.
#             When translate is False, this will be disabled.
#     """

#     # Compute how large the cropped image should be
#     if box_size is None:
#         box_size = int(max(boxj[3] - boxj[1], boxj[2] - boxj[0]))
#         box_size = \
#             int(min(max(box_size * 2, 128 * 1.2),
#                     375))
#         box_size = random.randrange(box_size, 376)

#     # Compute x, y coordinates in the image to serve as the box center.
#     # Offset the center so that the object is not always in the center
#     # (for data augmentation purposes)
#     box_center_x = (boxj[0] + boxj[2]) / 2
#     box_center_y = (boxj[1] + boxj[3]) / 2
#     if translate:
#         translation_offset = .5 * box_size / 2
#         box_center_x = int(random.uniform(
#             box_center_x - translation_offset,
#             box_center_x + translation_offset))
#         box_center_y = int(random.uniform(
#             box_center_y - translation_offset,
#             box_center_y + translation_offset))

#     # Compute (x1, y1) and (x2, y2)
#     x1 = int(box_center_x - (box_size / 2))
#     x2 = int(box_center_x + (box_size / 2))
#     y1 = int(box_center_y - (box_size / 2))
#     y2 = int(box_center_y + (box_size / 2))

#     shape = np.shape(image)

#     # Address out of bounds issues
#     if x1 < 0:
#         x2 -= x1
#         x1 = 0
#     if x2 >= shape[1]:
#         x1 -= x2 - shape[1]
#         x2 = shape[1]
#     if y1 < 0:
#         y2 -= y1
#         y1 = 0
#     if y2 >= shape[0]:
#         y1 -= y2 - shape[0]
#         y2 = shape[0]

#     return x1, x2, y1, y2
