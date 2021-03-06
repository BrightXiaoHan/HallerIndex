from src.contours_op import *
from src.utils import *
from .dicom_process import *
from easydict import EasyDict
from unet import cnt
from unet import cnt_min
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
        # 去掉横切面以外的图
        IOP = ds.ImageOrientationPatient
        IOP_round = [round(x) for x in IOP]
        if IOP_round != [1, 0, 0, 0, 1, 0]:
            return 0

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

    # 规则6 去掉脖子/纵切
    width = right_most[0] - left_most[0]
    height = bottom_most[1] - min(left_top[1], right_top[1])
    p = width/height
    if p > 2.2 or p < 1.35:
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
    
def analyse(dicom_file, _min):
    y = []
    xy = []

    dc = pydicom.dcmread(dicom_file)
    image = get_default_image(dc)
    if _min == False:
        inner_contour = cnt([image])[0]
    else:
        inner_contour = cnt_min([image])[0]
    # 找到左右胸轮廓的两个最低点，left_bottom是左侧，right_bottom是右侧
    left_chest_leftmost = find_boundary_point(inner_contour, position="left")
    right_chest_rightmost = find_boundary_point(inner_contour, position="right")

    cx = (left_chest_leftmost[0] + right_chest_rightmost[0]) / 2
    cy = (left_chest_leftmost[1] + right_chest_rightmost[1]) / 2

    left_inner_contour = filter_contour_points(inner_contour, x_max=cx)
    right_inner_contour = filter_contour_points(inner_contour, x_min=cx)

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
    threshold_value = y_max - 70
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

    r_yma = 0
    r_list = [[], []]
    for i in range(left_chest_leftmost[0], int(cx)):
        count_list = []
        for item in xy:
            if i == item[0]:
                count_list.append(item)
            if len(count_list) == 2:
                break
        try:
            y = abs(count_list[0][1] - count_list[1][1])
        except:
            continue
        if y > r_yma:
            r_yma = y
            r_list[0] = count_list[0]
            r_list[1] = count_list[1]

    l_yma = 0
    l_list = [[], []]
    for i in range(int(cx+50), right_chest_rightmost[0]):
        count_list = []
        for item in xy:
            if i == item[0]:
                count_list.append(item)
            if len(count_list) == 2:
                break
        try:
            y = abs(count_list[0][1] - count_list[1][1])
        except:
            continue
        if y > l_yma:
            l_yma = y
            l_list[0] = count_list[0]
            l_list[1] = count_list[1]

    left_inner_contour = filter_contour_points(inner_contour, x_max=cx)
    right_inner_contour = filter_contour_points(inner_contour, x_min=cx+50)

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
        "inner_contour": inner_contour,
        "left_top": left_top,
        "right_top": right_top,
        "left_bottom": left_bottom,
        "angle": angle,
        "r_list": r_list,
        "l_list": l_list,
        "r_yma": r_yma,
        "l_yma": l_yma,
        "mid_bottom": None  # 外轮廓中间凹陷点
    })
    
    return result_dict

def draw(dic, folder):
    """根据analyse函数提取的关键特征绘制辅助线并，计算Haller指数
    
    Args:
        result_dict (EasyDict): analyse函数的返回值
    
    Returns:
        EasyDict: "haller_index" Haller指数，"figure_image" 绘制辅助线之后的照片
    """
    r_list = dic.r_list
    l_list = dic.l_list
    r_yma = dic.r_yma
    l_yma = dic.l_yma
    asymmetry_index = r_yma / l_yma

    left_top = dic.left_top
    right_top = dic.right_top
    # mid_bottom = dic.mid_bottom
    #
    # left_y_distance = mid_bottom[1] - left_top[1]
    # right_y_distance = mid_bottom[1] - right_top[1]
    # angle = dic.angle
    # left_bottom = dic.left_bottom

    bottom_sternum_point = dic.bottom_sternum_point
    top_vertebra_point = dic.top_vertebra_point
    right_chest_rightmost = dic.right_chest_rightmost
    left_chest_leftmost = dic.left_chest_leftmost

    ab1 = bottom_sternum_point[1] - left_top[1]
    ab2 = bottom_sternum_point[1] - right_top[1]

    b = bottom_sternum_point[1] - top_vertebra_point[1]
    a = right_chest_rightmost[0] - left_chest_leftmost[0]

    haller_index = a / b

    # bottom_sternum_point = np.expand_dims(np.expand_dims(bottom_sternum_point, axis=0), axis=0)
    # top_vertebra_point = np.expand_dims(np.expand_dims(top_vertebra_point, axis=0), axis=0)
    # right_chest_rightmost = np.expand_dims(np.expand_dims(right_chest_rightmost, axis=0), axis=0)
    # left_chest_leftmost = np.expand_dims(np.expand_dims(left_chest_leftmost, axis=0), axis=0)
    #
    # matrix = cv2.getRotationMatrix2D((left_bottom[0], left_bottom[1]), -angle, 1.0)
    # bottom_sternum_point = rotate_contours(bottom_sternum_point, matrix)
    # top_vertebra_point = rotate_contours(top_vertebra_point, matrix)
    # right_chest_rightmost = rotate_contours(right_chest_rightmost, matrix)
    # left_chest_leftmost = rotate_contours(left_chest_leftmost, matrix)
    #
    # bottom_sternum_point = np.squeeze(bottom_sternum_point)
    # top_vertebra_point = np.squeeze(top_vertebra_point)
    # right_chest_rightmost = np.squeeze(right_chest_rightmost)
    # left_chest_leftmost = np.squeeze(left_chest_leftmost)

    inner_contour = dic.inner_contour
    # inner_contour = rotate_contours(inner_contour, matrix)
    x_list = []
    y_list = []
    l = len(inner_contour)
    for i in range(l):
        y_list.append(inner_contour[i][0][1])
        x_list.append(inner_contour[i][0][0])

    # show_contours(dic.img, dic.inner_contour)
    
    fig = plt.figure(figsize=(36, 36))
    image = dic.img
    # image = cv2.warpAffine(image, matrix, (image.shape[0], image.shape[1]))
    plt.imshow(image, cmap=plt.cm.gray)

    # 画出拟合曲线和原始点集
    # 画胸廓拟合点集
    plt.axis('equal')
    # 画外轮廓
    # plt.plot(out_contour[:, 0, 0], out_contour[:, 0, 1], color="black", linewidth=2)

    # 画左右连线
    y = (left_chest_leftmost[1] + right_chest_rightmost[1]) / 2
    xl = left_chest_leftmost[0]
    xr = right_chest_rightmost[0]

    plt.plot([r_list[0][0], r_list[1][0]], [r_list[0][1], r_list[1][1]], color="magenta", linewidth=4)
    plt.plot([l_list[0][0], l_list[1][0]], [l_list[0][1], l_list[1][1]], color="magenta", linewidth=4)
    plt.plot([xl, xr], [y, y], color="magenta", linewidth=4)
    # plt.plot([left_top[0], right_top[0]], [left_top[1], right_top[1]], color="magenta", linewidth=4)
    x = bottom_sternum_point[0]
    yt = top_vertebra_point[1]
    yb = bottom_sternum_point[1]

    xt = top_vertebra_point[0]
    plt.plot([xt+100, xt-100], [yb, yb], color="green", linewidth=4)
    plt.plot([xt+100, xt-100], [yt, yt], color="green", linewidth=4)
    # 画e 
    plt.plot([x, x], [yt, yb], color="cyan", linewidth=4)
    # 画内轮廓
    plt.scatter(x_list, y_list, c="b")

    # if left_y_distance >= right_y_distance:
    #     plt.plot([mid_bottom[0], mid_bottom[0]], [mid_bottom[1], left_top[1]], color="cyan", linewidth=4)
    # else:
    #     plt.plot([mid_bottom[0], mid_bottom[0]], [mid_bottom[1], right_top[1]], color="cyan", linewidth=4)

    plt.text(12, 12, "Width:%d, Hight:%d, Haller: %f." % (a, b, haller_index), fontsize=20, color="white")

    if ab1 > ab2:
        correction_index = (ab1 - b) / ab1
        plt.text(12, 30, "ab:%d, cd:%d, correction_index: %f%%." % (ab1, b, correction_index * 100), fontsize=20, color="white")
        plt.plot([left_top[0], left_top[0]], [left_top[1], yb], color="cyan", linewidth=4)
    else:
        correction_index = (ab2 - b) / ab2
        plt.text(12, 30, "ab:%d, cd:%d, correction_index: %f%%." % (ab2, b, correction_index * 100), fontsize=20, color="white")
        plt.plot([right_top[0], right_top[0]], [right_top[1], yb], color="cyan", linewidth=4)

    plt.text(12, 48, "R:%d, L:%d, asymmetry_index: %f." % (r_yma, l_yma, asymmetry_index), fontsize=20, color="white")

    plt.text(12, 70, "ID: {}.".format(folder), fontsize=20, color="white")

    figure_image = fig2img(fig)

    plt.close(fig)

    return haller_index, figure_image, a, correction_index, asymmetry_index


def degree_get(image):
    """拿到图片凹陷点和距离
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 没有找到对应的最凹陷点，为不可用图像
    try:
        contours = sorted(contours, key=lambda x: len(x))
        out_contour, out_contour_area = find_outer_contour(contours)

        out_contour, (cx, cy) = sort_clockwise(out_contour)

        left_top = find_boundary_point(filter_contour_points(out_contour, x_max=cx, y_max=cy), position="top")
        right_top = find_boundary_point(filter_contour_points(out_contour, x_min=cx, y_max=cy), position="top")

        mid_bottom = find_boundary_point(
            filter_contour_points(out_contour, x_min=left_top[0], x_max=right_top[0], y_max=cy), position="bottom")
    except Exception as e:
        return 0

    left_y_distance = mid_bottom[1] - left_top[1]
    right_y_distance = mid_bottom[1] - right_top[1]

    if left_y_distance >= right_y_distance:
        return left_y_distance, mid_bottom, left_top
    else:
        return right_y_distance, mid_bottom, right_top
