from src.contours_op import *
from src.utils import *
from .dicom_process import *
from easydict import EasyDict
from unet import cnt

def degree_of_depression(dicom_file):
    """判断一张ct图像的凹陷程度
    
    Args:
        dicom_file (str): dicom 文件的路径
    
    Returns:
        int: 凹陷程度值，即凹陷点到左右侧顶点连线的最大垂直距离。如果图像为不符合条件的图像，如运行错误，非横切胸片，则返回0
    """

    # DicomDir等文件运行此段代码会报错，为不可用图像
    try:
        ds = pydicom.read_file(dicom_file)
        img = cv2.convertScaleAbs(ds.pixel_array, alpha=(255.0/65535.0))
        ret, binary = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(
                        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        return 0

    # 没有找到对应的最凹陷点，为不可用图像
    try:
        contours = sorted(contours, key=lambda x: len(x))
        out_contour, out_contour_area = find_outer_contour(contours)

        out_contour, (cx, cy) = sort_clockwise(out_contour)

        left_top = find_boundary_point(filter_contour_points(out_contour,  x_max=cx, y_max=cy), position="top")
        right_top = find_boundary_point(filter_contour_points(out_contour, x_min=cx, y_max=cy), position="top")

        left_most = find_boundary_point(out_contour, position="left")
        right_most = find_boundary_point(out_contour, position="right")

        bottom_most = find_boundary_point(out_contour, position="bottom")
        mid_bottom = find_boundary_point(filter_contour_points(out_contour, x_min=left_top[0], x_max=right_top[0], y_max=cy), position="bottom")
    except Exception as e:
        return 0

    left_x_distance = mid_bottom[0] - left_top[0]
    right_x_distance = right_top[0] - mid_bottom[0]
    left_y_distance = mid_bottom[1] - left_top[1]
    right_y_distance = mid_bottom[1] - right_top[1]

    # 规则1  x轴距离差别过大
    if left_x_distance / right_x_distance > 3 or right_x_distance / left_x_distance > 3:
        return 0
    # 规则2 总体距离差距过小
    if left_x_distance < 10 or right_x_distance < 10:
        return 0

    # 规则2中点与两边的点y轴差距过小
    if left_y_distance < 15 or right_y_distance < 15:
        return 0
    
    # 规则3 底部点与中心店y轴距离过近的
    if cy - mid_bottom[1] < 10:
        return 0
    
    # 规则4 轮廓的宽大于高
    if bottom_most[1] - min(left_top[1], right_top[1]) > right_most[0] - left_most[0]:
        return 0

    # 规则5 轮廓中最左最右侧点与左右最高点的距离过小
    left_top_most_distance = left_top[0] - left_most[0]
    right_top_most_distance = right_most[0] - right_top[0]
    if left_x_distance / left_top_most_distance > 2 or right_x_distance / right_top_most_distance > 2:
        return 0
    
    # 计算凹陷点到左右两侧连线的距离
    degree = max(left_y_distance, right_y_distance)

    return degree

def segment(img):
    """从影像中分割出骨头轮廓以及组织轮廓。
    
    Args:
        img (np.ndarray): 通过pydicom.dcmread(file).pixel_array获得
    
    Returns:
        list: 组织轮廓
        list: 胸骨轮廓
    """
    # 阈值为3找到轮廓1
    contours_one = extract_contours_from_pxarray(img, 3)
    
    # 阈值为4找到轮廓2
    contours_two = extract_contours_from_pxarray(img, 4)
    
    # 阈值为5找到轮廓3
    contours_three = extract_contours_from_pxarray(img, 5)
    
    _, max_area_one = max_area_contour(contours_one)
    _, max_area_two = max_area_contour(contours_two)
    
    if abs(max_area_one - max_area_two) / max(max_area_one, max_area_two) > 0.6:
        return contours_one, contours_two
    else:
        return contours_two, contours_three


def diagnosis(dicom_file):
    return draw(analyse(dicom_file))
    
def analyse(dicom_file):
    y = []
    xy = []

    dc = pydicom.dcmread(dicom_file)
    image = get_default_image(dc)
    inner_contour = cnt([image])[0]

    # 找到左右胸轮廓的两个最低点，left_bottom是左侧，right_bottom是右侧
    left_chest_leftmost = find_boundary_point(inner_contour, position="left")
    right_chest_rightmost = find_boundary_point(inner_contour, position="right")

    cx = (left_chest_leftmost[0] + right_chest_rightmost[0]) / 2
    cy = (left_chest_leftmost[1] + right_chest_rightmost[1]) / 2

    left_inner_contour = filter_contour_points(inner_contour, x_max=cx)
    right_inner_contour = filter_contour_points(inner_contour, x_min=cy)

    left_bottom = find_boundary_point(left_inner_contour, position="bottom")
    right_bottom = find_boundary_point(right_inner_contour, position="bottom")

    # 以左侧最低点为中心，连线为轴将图像旋转，使得两点连线与X轴平行
    dy = right_bottom[1] - left_bottom[1]
    dx = right_bottom[0] - left_bottom[0]

    angle = np.arctan(dy / dx) / math.pi * 180

    if abs(angle) <= 15:
        # 旋转将胸廓ct摆正
        matrix = cv2.getRotationMatrix2D((left_bottom[0], left_bottom[1]), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (image.shape[0], image.shape[1]))
        inner_contour = rotate_contours(inner_contour, matrix)

    l = len(inner_contour)
    for i in range(l):
        y.append(inner_contour[i][0][1])
        xy.append(inner_contour[i][0])

    y_max = max(y)
    y_max = y_max - 10
    threshold_value = y_max - 50
    while y_max > min(y):
        count = 0
        for item in y:
            if item == y_max:
                count += 1
        if count > 2:
            y_max -= 10
        elif y_max >= threshold_value:
            y_max -= 10
        else:
            break

    left_chest_leftmost = find_boundary_point(inner_contour, position="left")
    right_chest_rightmost = find_boundary_point(inner_contour, position="right")
    
    cx = (left_chest_leftmost[0] + right_chest_rightmost[0])/2
    cy = (left_chest_leftmost[1] + right_chest_rightmost[1])/2
    
    left_inner_contour = filter_contour_points(inner_contour, x_max=cx)
    right_inner_contour = filter_contour_points(inner_contour, x_min=cx)

    left_top = find_boundary_point(left_inner_contour, position="top")
    right_top = find_boundary_point(right_inner_contour, position="top")
    
    vertebra = filter_contour_points(inner_contour, y_max=y_max, x_min=left_top[0], x_max=right_top[0])
    sternum = filter_contour_points(inner_contour, y_min=y_max, x_min=left_top[0], x_max=right_top[0])
    
    top_vertebra_point = find_boundary_point(vertebra, position="bottom")
    bottom_sternum_point = find_boundary_point(sternum, position="top")
    
    result_dict = EasyDict({
        "img": image,  # CT影像像素值
        "left_chest_leftmost": left_chest_leftmost,  # 左侧胸腔最外侧的点
        "right_chest_rightmost": right_chest_rightmost,  # 右侧胸腔最外侧的点
        "top_vertebra_point": top_vertebra_point,  # 胸肋骨最靠近胸腔的点（只包含中间部分）
        "bottom_sternum_point": bottom_sternum_point,  #  脊椎骨最靠近胸腔的点
        "vertebra": vertebra,  # 胸肋骨（中间部分）
        "sternum": sternum,  # 脊椎骨
        "inner_contour": inner_contour
    })
    
    return result_dict

def draw(dic):
    """根据analyse函数提取的关键特征绘制辅助线并，计算Haller指数
    
    Args:
        result_dict (EasyDict): analyse函数的返回值
    
    Returns:
        EasyDict: "haller_index" Haller指数，"figure_image" 绘制辅助线之后的照片
    """
    inner_contour = dic.inner_contour
    x_list = []
    y_list = []
    l = len(inner_contour)
    for i in range(l):
        y_list.append(inner_contour[i][0][1])
        x_list.append(inner_contour[i][0][0])

    b = dic.bottom_sternum_point[1] - dic.top_vertebra_point[1]
    a = dic.right_chest_rightmost[0] - dic.left_chest_leftmost[0]

    haller_index = a / b

    # show_contours(dic.img, dic.inner_contour)
    
    fig = plt.figure(figsize=(36, 36))
    plt.imshow(dic.img, cmap=plt.cm.gray)

    # 画出拟合曲线和原始点集
    # 画胸廓拟合点集
    plt.axis('equal')
    # 画外轮廓
    # plt.plot(out_contour[:, 0, 0], out_contour[:, 0, 1], color="black", linewidth=2)

    # 画左右连线
    y = (dic.left_chest_leftmost[1] + dic.right_chest_rightmost[1]) / 2
    xl = dic.left_chest_leftmost[0]
    xr = dic.right_chest_rightmost[0]

    plt.plot([xl, xr], [y, y], color="magenta", linewidth=4)

    x = dic.bottom_sternum_point[0]
    yt = dic.top_vertebra_point[1]
    yb = dic.bottom_sternum_point[1]

    xt = dic.top_vertebra_point[0]
    plt.plot([xt+80, xt-80], [yb, yb], color="green", linewidth=4)
    plt.plot([xt+80, xt-80], [yt, yt], color="green", linewidth=4)
    # 画e 
    plt.plot([x, x], [yt, yb], color="cyan", linewidth=4)
    # 画内轮廓
    plt.scatter(x_list, y_list, c="b")

    plt.text(24, 24, "Width:%d, Hight:%d, Haller: %f." % (a, b, haller_index), fontsize=50, color="white")

    figure_image = fig2img(fig)

    plt.close(fig)

    return haller_index, figure_image

