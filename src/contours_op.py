"""
与轮廓相关的操作函数
"""
import pydicom
import cv2
import math

import numpy as np
import pylab
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def find_inner_contour(contours, outline_area):
    """ 
    找到胸腔的内轮廓    
    Args:
        contours (list): 由长到短轮廓排序
        outline_area (float): 外胸廓的面积
    """

    # 存储所有符合条件的轮廓
    all_eligible_contour = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area / outline_area > 0.03 and area / outline_area < 0.5:
            all_eligible_contour.append(contour)

    if len(all_eligible_contour) < 2:
        return 0

    return all_eligible_contour[-2:]

def find_outer_contour(contours):
    """找到胸腔外部轮廓（轮廓）包围面积最大的为胸腔外轮廓
    
    Args:
        contours (list): 轮廓集合
    
    Returns:
        np.ndarray: 胸腔外轮廓, (n, 1, 2)
        float: 胸腔外轮廓面积
    """

    return max_area_contour(contours)


def show_contours(img, contours, copy=False, imshow=False, _imsave=None):
    """在指定图片的中展示指定的轮廓

    Args:
        img (numpy.ndarray): 原始图像
        contours (list): 轮廓集合
        copy (bool): True 在原图上绘制轮廓。False 创建图像的拷贝，在拷贝上绘制轮廓
        imshow (bool): 是否使用pyplot展示图像。如果你使用jupyter-notebook，可以将其设置为True
        _imsave (str): 加入此参数可以把生成的图片导出为本地文件
    Return:
        np.ndarray: 绘制后的图片矩阵
    """
    img_with_contours = np.copy(img) if copy else img
    cv2.drawContours(img_with_contours, contours, -1, (0, 100, 0), 3)
    if imshow:
        pylab.imshow(img_with_contours)
    if _imsave is not None:
        pylab.imsave(_imsave, img_with_contours, cmap=pylab.cm.gray)
    return img_with_contours
    
def show_points(img, points, copy=False, imshow=False):
    """在指定图片中绘制出指定的点
    
    Args:
        img ([type]): [description]
        points ([type]): [description]
        copy (bool): True 在原图上绘制轮廓。False 创建图像的拷贝，在拷贝上绘制轮廓
        imshow (bool): 是否使用pyplot展示图像。如果你使用jupyter-notebook，可以将其设置为True
    Return:
        np.ndarray: 绘制后的图片矩阵
    """
    img_with_points = np.copy(img) if copy else img
    for point in points:
        cv2.circle(img_with_points, tuple(point), 40, (0, 200, 0), 4)
    if imshow:
        pylab.imshow(img_with_points)
    return img_with_points

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

def max_area_contour(contours, diverse=False, filter_zero=False):
    """获取轮廓集合中面积最大的轮廓
    
    Args:
        contours (list): 轮廓集合
    """
    areas = []
    filtered_contours = []

    for c in contours:
        area = cv2.contourArea(c)
        # 过滤掉面积过小的轮廓
        if area < 5 and filter_zero:
            continue
        areas.append(area)
        filtered_contours.append(c)
    
    areas = np.array(areas)

    if diverse:
        index = areas.argmin()
    else:
        index = areas.argmax()
    return filtered_contours[index], areas[index]


def rotate_contours(contour, matrix):
    """旋转轮廓点坐标

    Args:
        contour (numpy.ndarray): shape of (n, 1, 2)
        matrix (numpy.ndarray): shape of (2, 3)
    """
    contour = np.squeeze(contour, 1).transpose(1, 0)
    pad = np.ones((1, contour.shape[1]))
    contour = np.concatenate([contour, pad])
    
    contour = np.dot(matrix, contour)
    contour = np.expand_dims(contour.transpose(1, 0), 1)
    return contour.astype(np.int)

def sort_clockwise(contour, center=None, anti=True, demarcation=0):
    """
    将轮廓坐标点逆时针排序

    Args:
        contour(np.ndarray): with shape (n, 1, 2)
        center(tuple): shape(2,).指定排序的中心点，如果未指定，则以轮廓的重心点为中心
        anti(bool): True为逆时针， False为顺时针
        demarcation(float): 排序的起始角度（用度数表示）
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

    angle = np.angle(plural, deg=True)

    angle = (angle + demarcation) % 360

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

def trap_contour(contour, img_shape, pixel=15):
    """将轮廓原地缩小若干个像素
    
    Args:
        contour (np.ndarray): 待缩小的轮廓 (n, 1, 2)
        img_shape (tuple): 原始图像的大小
        pixel (int, optional): 需要缩小的像素值. Defaults to 10.
    
    Returns:
        contour (np.ndarray): 缩小后的轮廓 (n, 1, 2)
    """
    img = np.ones(shape=img_shape, dtype="uint8") * 255
    cv2.drawContours(img, [contour], -1, (0, 0, 0), 3)
    dist = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    ring = cv2.inRange(dist, pixel-0.5, pixel+0.5) # take all pixels at distance between 9.5px and 10.5px

    _, contours, hierarchy = cv2.findContours(ring, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 面积最小的是内轮廓
    result_contour, area = max_area_contour(contours, diverse=True, filter_zero=True)

    return result_contour


def refine_contour(contour, img_shape):
    """重整轮廓，将轮廓点的内折擦除
    
    Args:
        contour (np.ndarray): 待缩小的轮廓 (n, 1, 2)
        img_shape (tuple): 原始图像的大小
    
    Returns:
        np.ndarray: 缩小后的轮廓
    """
    img = np.ones(shape=img_shape, dtype="uint8") * 255
    cv2.drawContours(img, [contour], -1, (0, 0, 0), -1)

    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    result_contour = sorted(contours, key=lambda x: len(x))[-1]

    return result_contour

def extract_contours_from_pxarray(pixel_array, threshold):
    """从dicom像素图片中提取轮廓点集
    
    Args:
        pixel_array (numpy.ndarray): 通过pydicom.dcmread(file).pixel_array获得, 并转换成uint8类型
        threshold (int): 提取轮廓的阈值
    
    Returns:
        list: list of contours
    """
    ret, binary = cv2.threshold(pixel_array, threshold, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours
    