import os
import re
import pydicom
import cv2
import math

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from PIL import Image
from scipy.interpolate import splev, splprep
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


def filter_contour_points(contour,  x_min=float('-inf'), x_max=float('inf'), y_min=float('-inf'), y_max=float('inf'), mode="keep"):
    """根据x，y轴的范围，过滤轮廓中的点，在此范围内的点才会被保留
    
    Args:
        contour (numpy.ndarray): shape (n, 1, 2)
        x_min (int): x轴坐标最小值
        x_max (int): x轴坐标最大值
        y_min (int): y轴坐标最小值
        y_max (int): y轴坐标最大值
        mode (str): keep,在此范围内的点会被保留， drop，在此范围外的点将被保留
    """
    if mode not in ["keep", "drop"]:
        raise AttributeError

    mask = np.ones((contour.shape[0],), dtype=bool)

    x = contour[:, 0, 0]
    y = contour[:, 0, 1]

    for m in [x < x_max, x > x_min, y < y_max, y > y_min]:
        mask = np.logical_and(mask, m)

    if mode == "drop":
        mask = np.logical_not(mask)

    return contour[mask]

def filter_contour_out_of_box(contour, contours):
    """保留所有点都在 "contour"轮廓指定范围内的轮廓
    
    Args:
        contour (numpy.ndarray): 目标轮廓，所有的点必须在这个轮廓里
        contours (numpy.ndarray): 需要过滤的轮廓集合
    """
    result = []

    for c in contours:
        flag = True
        for p in c:
            if not cv2.pointPolygonTest(contour, p, False):
                flag = False
                break
        if flag:
            result.append(c)
    
    return result


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

def sort_clockwise(contour, center=None, anti=True):
    """
    将轮廓坐标点逆时针排序

    Args:
        contour(np.ndarray): with shape (n, 1, 2)
        center(tuple): shape(2,).指定排序的中心点，如果未指定，则以轮廓的重心点为中心
        anti(bool): True为逆时针， False为顺时针
    Return:
        np.ndarray: with shape (n, 1, 2)
    """
    # 计算轮廓的中心
    if center is None:
        M=cv2.moments(contour) 
        cx=int(M['m10']/M['m00'])
        cy=int(M['m01']/M['m00'])
    else:
        assert len(center) == 2, "center参数必须为长度为2的tuple或者list。请检查您的参数"
        cx, cy = center

    x = contour[:, 0, 0]
    y = contour[:, 0, 1]

    plural = (x - cx) + (y - cy) * 1j

    angle = np.angle(plural)

    if anti:  # 逆时针排序
        sort_keys = np.argsort(angle)
    else:  # 顺时针排序
        sort_keys = np.argsort(-angle)

    result = np.expand_dims(np.stack([x[sort_keys], y[sort_keys]], 1), 1)

    return result, (cx, cy)

def nearest_point(contour, point):
    """找到轮廓中距离目标点最近的点
    
    Args:
        contour (numpy.ndarray): shape (n, 1, 2)
        point (numpy.ndarray): shape (2,)
    """
    x = contour[:, 0, 0]
    y = contour[:, 0, 1]

    distance = np.sqrt((x - point[0])**2 + (y - point[1])**2)
    sort_keys = np.argsort(distance)

    result = np.array([x[sort_keys[0]], y[sort_keys[0]]])
    return result


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

    # 找到胸外轮廓
    out_contour, _ = sort_clockwise(contours[-1])

    # 找到外胸轮廓的最高点和最低点
    out_contour_bottom = find_boundary_point(out_contour, "bottom")
    out_contour_top = find_boundary_point(out_contour, "top")

    # 找到内胸腔轮廓
    inner_contours = find_inner_contour(contours)

    # 找到左右胸轮廓的两个最低点，lowest_1是左侧，lowest_2是右侧
    lowest_1 = find_boundary_point(inner_contours[0], position="bottom")
    lowest_2 = find_boundary_point(inner_contours[1], position="bottom")

    # 交换位置 1 是左胸，2 是右胸
    inner_contours[0], inner_contours[1] = (inner_contours[0], inner_contours[1]) if lowest_1[0] < lowest_2[0] else (
        inner_contours[1], inner_contours[0])
    inner_contours[0], _ = sort_clockwise(inner_contours[0])
    inner_contours[1], _ = sort_clockwise(inner_contours[1])

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
        rib_contours, x_min=lowest_1[0], x_max=lowest_2[0], mode='all') 

    # 取左右最外侧点的中点为上下胸分界点
    demarcation_point = (left_chest_leftmost[1] + right_chest_rightmost[1]) / 2

    # 以此分界点为接线，将胸骨分为上下两个部分
    top_rib_contours = filter_contours(rib_contours, y_max=demarcation_point, y_min=out_contour_top[1], mode="all")
    bottom_rib_contours = filter_contours(rib_contours, y_min=demarcation_point, y_max=out_contour_bottom[1], mode="all")

    if len(top_rib_contours) == 0 or len(bottom_rib_contours) == 0:
        raise SternumVertebraNotFoundException("请检查您的图像是否符合要求，自动检测无法找找到胸骨。")

    # 将上下胸骨的轮廓合并
    vertebra_contour = top_rib_contours[-1]
    sternum_contour = np.concatenate(bottom_rib_contours)

    # 寻找脊椎骨最上点， 和胸骨最下点
    top_vertebra_point = find_boundary_point(vertebra_contour, "bottom")
    bottom_sternum_point = find_boundary_point(sternum_contour, "top")

    b = bottom_sternum_point[1] - top_vertebra_point[1]

    haller_index = a / b

    # 找到脊椎骨的左右侧点，和距离左右胸最近的点
    vertebra_contour_left_most = find_boundary_point(vertebra_contour, position="left")
    vertebra_contour_right_most = find_boundary_point(vertebra_contour, position="right")
    left_chest_near_vertebra = nearest_point(inner_contours[0], vertebra_contour_left_most)
    right_chest_near_vertebra = nearest_point(inner_contours[1], vertebra_contour_right_most)

    # 找到胸骨的顶点， 和距离胸骨定点最近的左右胸轮廓点
    left_chest_near_sternum = nearest_point(inner_contours[0], bottom_sternum_point)
    right_chest_near_sternum = nearest_point(inner_contours[1], bottom_sternum_point)

    # 过滤左右胸的点
    inner_contours[0] = filter_contour_points(inner_contours[0], 
                                              x_min=lowest_1[0], 
                                              y_max=left_chest_near_sternum[1],
                                              y_min=left_chest_near_vertebra[1],
                                              mode="drop")
    # 过滤右胸的点
    inner_contours[1] = filter_contour_points(inner_contours[1], 
                                              x_max=lowest_2[0], 
                                              y_min=right_chest_near_vertebra[1],
                                              y_max=right_chest_near_sternum[1],
                                              mode="drop")

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(img * 20)

    # 画出拟合曲线和原始点集
    # 画胸廓拟合点集
    plt.axis('equal')
    # 画外轮廓
    plt.plot(out_contour[:, 0, 0], out_contour[:, 0, 1], color="black", linewidth=2)

    # 画内轮廓
    plt.scatter(inner_contours[0][:, 0, 0], inner_contours[0][:, 0, 1], color="black", linewidth=2)
    plt.scatter(inner_contours[1][:, 0, 0], inner_contours[1][:, 0, 1], color="black", linewidth=2)

    # 闭合内外轮廓曲线
    plt.plot([left_chest_near_vertebra[0], right_chest_near_vertebra[0]],
            [left_chest_near_vertebra[1], right_chest_near_vertebra[1]],
            color="black", 
            linewidth=4
    
    )
    plt.plot([left_chest_near_sternum[0], bottom_sternum_point[0], right_chest_near_sternum[0]],
            [left_chest_near_sternum[1], bottom_sternum_point[1], right_chest_near_sternum[1]],
            color="black", 
            linewidth=4
    )
    
    # 画上胸骨
    plt.scatter(sternum_contour[:, 0, 0], sternum_contour[:, 0, 1], color="black", linewidth=1)

    # 画脊椎骨
    plt.scatter(vertebra_contour[:, 0, 0], vertebra_contour[:, 0, 1], color="black", linewidth=1)
    
    # 画左右连线
    plt.plot(*zip(*[left_chest_leftmost, right_chest_rightmost]), color="magenta", linewidth=2)

    # 画e 
    plt.plot(*zip(*[top_vertebra_point, bottom_sternum_point]), color="cyan", linewidth=2)

    plt.text(out_contour_top[0], out_contour_top[1] - 24, "Width:%d, Hight:%d, Haller: %f." % (a, b, haller_index), fontsize=10, color="white")

    for c in rib_contours:
        plt.scatter(c[:, 0, 0], c[:, 0, 1], color="yellow", linewidth=1)

    plt.legend()

    figure_image = fig2img(fig)
    
    # 如果需要绘制相应的分析图像，输出到指定文件
    if saved_path is not None:
        plt.savefig(saved_path)
    
    return haller_index, figure_image