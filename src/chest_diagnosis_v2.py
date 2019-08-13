import os
import re
import pydicom
import cv2
import math

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from PIL import Image
from src.utils import fig2img

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


def show_contours(img, contours):
    """在坐标轴中展示指定的轮廓

    Args:
        img (numpy.ndarray): 原始图像
        contours (list): 轮廓集合
    """
    img_with_contours = np.copy(img)
    cv2.drawContours(img_with_contours, contours, -1, (255, 255, 255), 3)
    # plt.imshow(img_with_contours, cmap=plt.cm.bone)
    plt.imsave("hello.jpg", img_with_contours, cmap=plt.cm.bone)


def find_boundary_point(contour, position):
    """找到指定轮廓的最低点，并返回, （注意绘图时x轴坐标由左至右递增， y轴坐标由上至下递增）

    Args:
        contour (numpy.ndarray): shape (n, 1, 2)
        position (str): ["bottom", "top", "left", "right"] 
    """
    if position not in ["bottom", "top", "left", "right"]:
        raise AttributeError(
            'Position 参数必须是  ["bottom", "top", "left", "right"] 其中之一')

    if position in ["bottom", "right"]:
        func = np.argmax
    else:
        func = np.argmin

    if position in ["bottom", "top"]:
        axis_index = 1
    else:
        axis_index = 0

    index = func(contour[:, 0, axis_index])
    point = contour[index, 0]

    return point


def filter_contours(contours, x_min=float('-inf'), x_max=float('inf'), y_min=float('-inf'), y_max=float('inf'), mode="exist"):
    """根据x，y轴的范围过滤轮廓，只保留存在点集在这个范围内的轮廓

    Args:
        contours (list): 轮廓点列表，每个元素是numpy.ndarray类型 (n, 1, 2)
        x_min (int): x轴坐标最小值
        x_max (int): x轴坐标最大值
        y_min (int): y轴坐标最小值
        y_max (int): y轴坐标最大值
        mode (str): ["exist", "all"]其中之一。exist模式表示存在满足条件的点即保存该轮廓，all模式表示所有的点满足条件才保存该轮廓。
    """
    result_contours = []
    for contour in contours:
        x = contour[:, 0, 0]
        y = contour[:, 0, 1]

        if mode == "exist":
            if np.sum(np.logical_and(x < x_max, x > x_min)) > 0 and np.sum(np.logical_and(y < y_max, y > y_min)) > 0:
                result_contours.append(contour)
        elif mode == "all":
            if np.sum(np.logical_or(x > x_max, x < x_min)) == 0 and np.sum(np.logical_or(y > y_max, y < y_min)) == 0:
                result_contours.append(contour)
        else:
            raise AttributeError('请指定mode参数为["exist", "all"]其中之一')

    return result_contours


def filter_contour_points(contour,  x_min=float('-inf'), x_max=float('inf'), y_min=float('-inf'), y_max=float('inf')):
    """根据x，y轴的范围，过滤轮廓中的点，在此范围内的点才会被保留
    
    Args:
        contour (numpy.ndarray): shape (n, 1, 2)
        x_min (int): x轴坐标最小值
        x_max (int): x轴坐标最大值
        y_min (int): y轴坐标最小值
        y_max (int): y轴坐标最大值
    """
    mask = np.ones((contour.shape[0],), dtype=bool)

    x = contour[:, 1, 0]
    y = contour[:, 1, 1]

    for m in [x < x_max, x > x_min, y < x_max, y > y_min]:
        mask = np.logical_and(mask, m)

    return contour[mask]


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


class SternumVertebraNotFoundException(Exception):
    """无法找到脊椎骨或胸骨的错误
    """
    pass


def diagnosis(dicom_file, saved_path=None):

    # 读取dicom文件中的像素数据
    ds = pydicom.dcmread(dicom_file)
    img = cv2.convertScaleAbs(ds.pixel_array, alpha=(255.0/65535.0))

    # 提取像素轮廓点
    ret, binary = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 将所有轮廓按轮廓点数量由大到小排序
    contours = sorted(contours, key=lambda x: len(x))

    out_contour = contours[-1]
    inner_contours = find_inner_contour(contours)

    # 找到左右胸轮廓的两个最低点，lowest_1是左侧，lowest_2是右侧
    lowest_1 = find_boundary_point(inner_contours[0], position="bottom")
    lowest_2 = find_boundary_point(inner_contours[1], position="bottom")

    # 交换位置 1 是左胸，2 是右胸
    inner_contours[0], inner_contours[1] = (inner_contours[0], inner_contours[1]) if lowest_1[0] < lowest_2[0] else (
        inner_contours[1], inner_contours[0])
    lowest_1, lowest_2 = (lowest_1, lowest_2) if lowest_1[0] < lowest_2[0] else (
        lowest_2, lowest_1)

    # 以左侧最低点为中心，连线为轴将图像旋转，使得两点连线与X轴平行
    dy = lowest_2[1] - lowest_1[1]
    dx = lowest_2[0] - lowest_1[0]

    angle = np.arctan(dy / dx) / math.pi * 180

    matrix = cv2.getRotationMatrix2D((lowest_1[0], lowest_1[1]), angle, 1.0)
    img = cv2.warpAffine(img, matrix, (img.shape[0], img.shape[1]))

    inner_contours = [rotate_contours(contour, matrix)
                      for contour in inner_contours]

    # 找到左右胸最外侧的点，计算a
    left_chest_leftmost = find_boundary_point(
        inner_contours[0], position="left")
    right_chest_rightmost = find_boundary_point(
        inner_contours[1], position="right")

    a = right_chest_rightmost[0] - left_chest_leftmost[0]

    # 提取胸骨轮廓点
    ret, binary = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    _, rib_contours, _ = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rib_contours = sorted(rib_contours, key=lambda x: len(x))


    # 找到左右胸轮廓最靠近中间的点
    left_chest_rightmost = find_boundary_point(
        inner_contours[0], position="right")
    right_chest_leftmost = find_boundary_point(
        inner_contours[1], position="left")

    rib_contours = filter_contours(
        rib_contours, x_min=left_chest_rightmost[0], x_max=right_chest_leftmost[0]) 

    # 取左右最外侧点的中点为上下胸分界点
    demarcation_point = (left_chest_leftmost[1] + right_chest_rightmost[1]) / 2

    # 以此分界点为接线，将胸骨分为上下两个部分
    top_rib_contours = filter_contours(rib_contours, y_max=demarcation_point, mode="all")
    bottom_rib_contours = filter_contours(rib_contours, y_min=demarcation_point, mode="all")

    if len(top_rib_contours) == 0 or len(bottom_rib_contours) == 0:
        raise SternumVertebraNotFoundException("请检查您的图像是否符合要求，自动检测无法找找到胸骨。")

    # 取上下胸骨最大的轮廓作为脊椎骨和胸骨
    vertebra_contour = top_rib_contours[-1]
    sternum_contour = bottom_rib_contours[-1]

    # 寻找脊椎骨最上点， 和胸骨最下点
    top_vertebra_point = find_boundary_point(vertebra_contour, "bottom")
    bottom_sternum_point = find_boundary_point(sternum_contour, "top")

    b = bottom_sternum_point[1] - top_vertebra_point[1]

    haller_index = a / b

    fig = plt.figure(figsize=(16, 6))
    # -------------------------------------------- #
    # 此处画第一张子图                                #
    # -------------------------------------------- #
    plt.subplot(121)
    plt.imshow(img)

    # -------------------------------------------- #
    # 此处画第二张子图                                #
    # -------------------------------------------- #
    plt.subplot(122)
    # 画出拟合曲线和原始点集
    # 画胸廓拟合点集
    plt.axis('equal')
    # 画外轮廓
    plt.scatter(out_contour[:, 0, 0], out_contour[:, 0, 1], color="black", label="Outline", linewidth=2)

    # 画内轮廓
    plt.scatter(inner_contours[0][:, 0, 0], inner_contours[0][:, 0, 1], color="black", label="Innerline", linewidth=2)
    plt.scatter(inner_contours[1][:, 0, 0], inner_contours[1][:, 0, 1], color="black", label="Innerline", linewidth=2)
    
    # 画上胸骨
    plt.scatter(sternum_contour[:, 0, 0], sternum_contour[:, 0, 1], color="black", label="sternum_contour", linewidth=2)

    # 画脊椎骨
    plt.scatter(vertebra_contour[:, 0, 0], vertebra_contour[:, 0, 1], color="black", label="vertebra_contour", linewidth=2)
    
    # 画左右连线
    plt.plot(*zip(*[left_chest_leftmost, right_chest_rightmost]), color="magenta", label="a", linewidth=2)

    # 画e 
    plt.plot(*zip(*[top_vertebra_point, bottom_sternum_point]), color="cyan", label="b", linewidth=2)

    plt.legend()

    figure_image = fig2img(fig)
    
    # 如果需要绘制相应的分析图像，输出到指定文件
    if saved_path is not None:
        plt.savefig(saved_path)
    
    return haller_index, figure_image