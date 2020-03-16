from src.contours_op import *
from src.utils import *
from .dicom_process import *
from unet import cnt


def degree_of_inner(dicom_file):
    y = []
    xy = []

    dc = pydicom.dcmread(dicom_file)
    image = get_default_image(dc)

    try:
        inner_contour = cnt([image])[0]
    except:
        return 0

    # 找到左右胸轮廓的两个最低点，left_bottom是左侧，right_bottom是右侧
    left_chest_leftmost = find_boundary_point(inner_contour, position="left")
    right_chest_rightmost = find_boundary_point(inner_contour, position="right")

    cx = (left_chest_leftmost[0] + right_chest_rightmost[0]) / 2

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

    cx = (left_chest_leftmost[0] + right_chest_rightmost[0]) / 2

    left_inner_contour = filter_contour_points(inner_contour, x_max=cx)
    right_inner_contour = filter_contour_points(inner_contour, x_min=cx)

    left_top = find_boundary_point(left_inner_contour, position="top")
    right_top = find_boundary_point(right_inner_contour, position="top")

    vertebra = filter_contour_points(inner_contour, y_max=y_max, x_min=left_top[0], x_max=right_top[0])

    try:
        top_vertebra_point = find_boundary_point(vertebra, position="bottom")
    except:
        return 0

    left_y_distance = top_vertebra_point[1] - left_top[1]
    right_y_distance = top_vertebra_point[1] - right_top[1]

    degree = max(left_y_distance, right_y_distance)

    return degree
