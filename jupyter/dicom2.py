# %%
# 导入依赖包
import os
import re
import pydicom
import cv2
import math

import numpy as np
import matplotlib.pyplot as plt


# %%
# 读取dicom文件
folder = 'data/pe'
f = os.listdir(folder)[9]
print(f)
ds = pydicom.dcmread(os.path.join(folder, f))  # plan dataset
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
img = cv2.convertScaleAbs(ds.pixel_array, alpha=(255.0/65535.0))
plt.imshow(img, cmap=plt.cm.bone)


# %%
# 提取像素轮廓点
ret, binary = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)
_, contours, _ = cv2.findContours(
    binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 将所有轮廓按轮廓点数量由大到小排序
contours = sorted(contours, key=lambda x: len(x))


def find_inner_contour(contours):
    """ 
    找到胸腔的内轮廓    
    Args:
        contours (list): 由长到短轮廓排序
    """
    # 首先取出胸腔的外轮廓
    outline_contour = contours[-1]
    outline_area = cv2.contourArea(outline_contour)

    # 存储所有符合条件的轮廓
    all_eligible_contour = []

    for contour in contours[:-1]:
        area = cv2.contourArea(contour)
        if area / outline_area > 0.1:
            all_eligible_contour.append(contour)

    if len(all_eligible_contour) < 2:
        raise IndexError(
            "Please check the image you given. Can't find inner contour of chest.")

    return all_eligible_contour[-2:]


inner_contours = find_inner_contour(contours)


def show_contours(img, contours):
    """在坐标轴中展示指定的轮廓

    Args:
        img (numpy.ndarray): 原始图像
        contours (list): 轮廓集合
    """
    img_with_contours = np.copy(img)
    cv2.drawContours(img_with_contours, contours, -1, (255, 255, 255), 3)
    plt.imshow(img_with_contours, cmap=plt.cm.bone)


show_contours(img, inner_contours)
# %%
# 纠正内轮廓的方向


def find_lowest_point(contour):
    """找到指定轮廓的最低点，并返回

    Args:
        contour (numpy.ndarray): shape (n, 1, 2) 
    """

    min_index = np.argmax(contour[:, 0, 1])
    lowest = contour[min_index, 0]

    return lowest


# 找到左右胸轮廓的两个最低点，lowest_1是左侧，lowest_2是右侧
lowest_1 = find_lowest_point(inner_contours[0])
lowest_2 = find_lowest_point(inner_contours[1])

lowest_1, lowest_2 = (lowest_1, lowest_2) if lowest_1[0] < lowest_2[0] else (
    lowest_2, lowest_1)


# 以左侧最低点为中心，连线为轴将图像旋转，使得两点连线与X轴平行
dy = lowest_2[1] - lowest_1[1]
dx = lowest_2[0] - lowest_1[0]

angle = np.arctan(dy / dx) / math.pi * 180

matrix = cv2.getRotationMatrix2D((lowest_1[0], lowest_1[1]), angle, 1.0)
img = cv2.warpAffine(img, matrix, (img.shape[0], img.shape[1]))
plt.imshow(img, cmap=plt.cm.bone)

# 旋转轮廓点


def rotate_contours(contour, matrix):
    """旋转轮廓点坐标

    Args:
        contour (numpy.ndarray): shape of (n, 1, 2)
        matrix (numpy.ndarray): shape of (2, 3)
    """
    contour = np.squeeze(contour).transpose(1, 0)
    pad = np.ones((1, contour.shape[1]))
    contour = np.concatenate([contour, pad])
    contour = np.dot(matrix, contour)
    contour = np.expand_dims(contour.transpose(1, 0), 1)
    return contour.astype(np.int)


inner_contours = [rotate_contours(contour, matrix)
                  for contour in inner_contours]
show_contours(img, inner_contours)


# %%

# %%
# 提取胸骨轮廓点
ret, binary = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
_, rib_contours, _ = cv2.findContours(
    binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
rib_contours = sorted(rib_contours, key=lambda x: len(x))


# 轮廓最大的事脊椎骨的轮廓
sternum_contour = rib_contours[-1]

# 找出脊椎骨轮廓的最内点, 最左侧点，和最右侧点
top_sternum_point_index = np.argmin(sternum_contour[:, 0, 1])
top_sternum_point = sternum_contour[top_sternum_point_index, 0]

left_sternum_point_index = np.argmin(sternum_contour[:, 0, 0])
left_sternum_point = sternum_contour[left_sternum_point_index, 0]

right_sternum_point_index = np.argmax(sternum_contour[:, 0, 0])
right_sternum_point = sternum_contour[right_sternum_point_index, 0]

# 找出胸骨的位置
vertebra_contours = []
for contour in rib_contours[:-1]:
    x = contour[:, 0, 0]
    y = contour[:, 0, 1]
    if np.sum(np.logical_and(x < right_sternum_point[0], x > left_sternum_point[0])) > 0 and np.sum(y < top_sternum_point[0]) >= 0:
        vertebra_contours.append(contour)

if len(vertebra_contours) == 0:
    raise Exception("Vertebra is not found in this diagram.")

vertebra_contour = vertebra_contours[-1]
bottom_vertebra_point_index = np.argmax(vertebra_contour[:, 0, 1])
bottom_vertebra_point = vertebra_contour[bottom_vertebra_point_index, 0]

show_contours(img, [sternum_contour, vertebra_contour])
# %%
all_useful_contours = []
all_useful_contours.extend(inner_contours)
all_useful_contours.extend([sternum_contour, vertebra_contour])
show_contours(img, all_useful_contours)

#%%
