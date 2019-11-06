from src.contours_op import *
from src.utils import *

def degree_of_depression(dicom_file):
    """判断一张ct图像的凹陷程度
    
    Args:
        dicom_file (str): dicom 文件的路径
    
    Returns:
        int: 凹陷程度值，即凹陷点到左右侧顶点连线的最大垂直距离。如果图像为不符合条件的图像，如运行错误，非横切胸片，则返回0
    """
    ds = pydicom.read_file(dicom_file)

    # DicomDir等文件运行此段代码会报错，为不可用图像
    try:
        img = cv2.convertScaleAbs(ds.pixel_array, alpha=(255.0/65535.0))
    except:
        return 0

    ret, binary = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: len(x))
    out_contour, out_contour_area = find_outer_contour(contours)

    out_contour, (cx, cy) = sort_clockwise(out_contour)

    left_top = find_boundary_point(filter_contour_points(out_contour,  x_max=cx, y_max=cy), position="top")
    right_top = find_boundary_point(filter_contour_points(out_contour, x_min=cx, y_max=cy), position="top")

    left_most = find_boundary_point(out_contour, position="left")
    right_most = find_boundary_point(out_contour, position="right")

    # 没有找到中间最凹陷点，为不可用图像
    try:
        mid_bottom = find_boundary_point(filter_contour_points(out_contour, x_min=left_top[0], x_max=right_top[0], y_max=cy), position="bottom")
    except ValueError as e:
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
    if left_y_distance < 10 or right_y_distance < 10:
        return 0
    
    # 规则3 底部点与中心店y轴距离过近的
    if cy - mid_bottom[1] < 10:
        return 0

    # 规则4 轮廓中最左最右侧点与左右最高点的距离过小
    left_top_most_distance = left_top[0] - left_most[0]
    right_top_most_distance = right_most[0] - right_top[0]
    if left_x_distance / left_top_most_distance > 2 or right_x_distance / right_top_most_distance > 2:
        return 0
    
    # 计算凹陷点到左右两侧连线的距离
    degree = max(left_y_distance, right_y_distance)

    return degree

def diagnosis(dicom_file, plot=True):
    """计算给定胸部横切照片的Haller指数
    
    Args:
        dicom_file (str): 胸部横切dicom文件
        plot (bool): 是否在原图上绘制辅助线
    
    Returns:
        tuple: haller_index (Haller指数), figure_image(带辅助线的照片) 注：如果plot为Fasle, 将只返回Haller指数
    """
    # ------------------------------------------------------------------------- #
    #        读取dicom文件中的像素数据                                             
    # ------------------------------------------------------------------------- #
  
    ds = pydicom.dcmread(dicom_file)
    img = cv2.convertScaleAbs(ds.pixel_array, alpha=(255.0/65535.0))
    origin_img = cv2.convertScaleAbs(ds.pixel_array, alpha=(2550.0/65535.0))

    # ------------------------------------------------------------------------- #
    #        使用阈值为3提取像素轮廓点                                             
    # ------------------------------------------------------------------------- #
    ret, binary = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 将所有轮廓按轮廓点数量由大到小排序
    contours = sorted(contours, key=lambda x: len(x))


    # ------------------------------------------------------------------------- #
    #        找外胸腔轮廓及其关键点                                           
    # ------------------------------------------------------------------------- #
    # 找到胸外轮廓(区域面积最大的为外胸廓轮廓点)
    out_contour, out_contour_area = find_outer_contour(contours)
    out_contour, (cx, cy) = sort_clockwise(out_contour)

    # 找到外胸廓突点，和外轮廓凹点
    left_top = find_boundary_point(filter_contour_points(out_contour, x_max=cx, y_max=cy), position="top")
    right_top = find_boundary_point(filter_contour_points(out_contour, x_min=cx, y_max=cy), position="top")

    mid_bottom = find_boundary_point(filter_contour_points(out_contour, x_min=left_top[0], x_max=right_top[0], y_max=cy), position="bottom")

    # 找到外胸轮廓的最高点和最低点
    out_contour_bottom = find_boundary_point(out_contour, "bottom")
    out_contour_top = find_boundary_point(out_contour, "top")

    # ------------------------------------------------------------------------- #
    #        找内胸腔轮廓及其关键点                                           
    # ------------------------------------------------------------------------- #
    # 找到内胸腔轮廓
    inner_contours = find_inner_contour(contours, out_contour_area)

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

    # ------------------------------------------------------------------------- #
    #        将图像及其轮廓旋转（将其摆正，使其水平）                                           
    # ------------------------------------------------------------------------- #
    # 以左侧最低点为中心，连线为轴将图像旋转，使得两点连线与X轴平行
    dy = lowest_2[1] - lowest_1[1]
    dx = lowest_2[0] - lowest_1[0]

    angle = np.arctan(dy / dx) / math.pi * 180

    # 旋转将胸廓ct摆正
    matrix = cv2.getRotationMatrix2D((lowest_1[0], lowest_1[1]), angle, 1.0)
    img = cv2.warpAffine(img, matrix, (img.shape[0], img.shape[1]))
    origin_img = cv2.warpAffine(origin_img, matrix, (img.shape[0], img.shape[1]))

    inner_contours = [rotate_contours(contour, matrix)
                        for contour in inner_contours]
    out_contour = rotate_contours(out_contour, matrix)

    inner_left_top_point = find_boundary_point(inner_contours[0], "top")
    inner_right_top_point = find_boundary_point(inner_contours[1], "top")

    # ------------------------------------------------------------------------- #
    #        找到左右胸最外侧的点，计算a（即左右内胸腔边界连线）                                           
    # ------------------------------------------------------------------------- # 
    left_chest_leftmost = find_boundary_point(
        inner_contours[0], position="left")
    right_chest_rightmost = find_boundary_point(
        inner_contours[1], position="right")

    # ------------------------------------------------------------------------- #
    #        使用阈值为4，提取胸骨轮廓点                                        
    # ------------------------------------------------------------------------- #
    ret, binary = cv2.threshold(img, 4, 255, cv2.THRESH_BINARY)
    _, rib_contours, _ = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rib_contours = sorted(rib_contours, key=lambda x: len(x))
    rib_contours_all_in_one = np.concatenate(rib_contours)

    # ------------------------------------------------------------------------- #
    #        找脊椎骨与胸肋骨轮廓以及关键点  vertebra：胸肋骨   sternum：脊椎骨                                  
    # ------------------------------------------------------------------------- #
    # 找到左右胸轮廓最靠近中间的点
    left_chest_rightmost = find_boundary_point(
        inner_contours[0], position="right")
    right_chest_leftmost = find_boundary_point(
        inner_contours[1], position="left")

    rib_contours = filter_contours(
        rib_contours, x_min=lowest_1[0], x_max=lowest_2[0], mode='all') 

    # 取左右最外侧点的中点为上下胸分界点
    demarcation_point = (left_chest_leftmost[1] + right_chest_rightmost[1]) / 2 - 20  # 由于有的胸骨轮廓会超过中点线， 所以此处以重点线上方10像素为分界点

    # 以此分界点为接线，将胸骨分为上下两个部分
    bottom_rib_contours = filter_contours(rib_contours, y_min=demarcation_point, y_max=out_contour_bottom[1], x_min=left_chest_leftmost[0], x_max=right_chest_rightmost[0], mode="all")

    # # 下胸骨选轮廓集合的top3
    # if len(bottom_rib_contours) >= 3:
    #     bottom_rib_contours = bottom_rib_contours[-3:]

    # 外胸廓凹陷点向下作为胸肋骨点
    trapped_outter_contour = trap_contour(out_contour, img.shape)
    tmp_points = find_boundary_point(filter_contour_points(trapped_outter_contour, x_min=left_top[0], x_max=right_top[0], y_max=cy), position="bottom")

    # 将上下胸骨的轮廓合并
    vertebra_contour = np.concatenate(filter_contours(rib_contours, y_max=tmp_points[1] + 20, y_min=mid_bottom[1], mode="all"))
    sternum_contour = np.concatenate(bottom_rib_contours)

    # 寻找脊椎骨最上点， 和胸骨最下点
    top_vertebra_point = find_boundary_point(vertebra_contour, "bottom")
    bottom_sternum_point = find_boundary_point(sternum_contour, "top")

    # 寻找环绕胸骨的最左侧点和最右侧点
    rib_contours_all_in_one = filter_contour_points(trapped_outter_contour, y_min=top_vertebra_point[1], y_max=bottom_sternum_point[1])
    left_rib_point = find_boundary_point(rib_contours_all_in_one, "left")
    left_rib_point[0] = left_rib_point[0] + 15
    right_rib_point = find_boundary_point(rib_contours_all_in_one, "right")
    right_rib_point[0] = right_rib_point[0] - 15
    # ------------------------------------------------------------------------- #
    #        计算b，即内胸廓凹陷点与脊椎骨上侧点的连线                                 
    # ------------------------------------------------------------------------- #    
    b = bottom_sternum_point[1] - top_vertebra_point[1]
    # 如果左右x轴相差过大，则使用胸骨点作为左右连线
    if abs(left_chest_leftmost[1] - right_chest_rightmost[1]) > 20: 
        left_chest_leftmost = left_rib_point
        right_chest_rightmost = right_rib_point

    a = right_chest_rightmost[0] - left_chest_leftmost[0]

    haller_index = a / b

    if not plot:
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(origin_img)
        figure_image = fig2img(fig)
        return haller_index, figure_image

    # ------------------------------------------------------------------------- #
    #       绘制辅助线                                  
    # ------------------------------------------------------------------------- #
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(origin_img)

    # 画出拟合曲线和原始点集
    # 画胸廓拟合点集
    plt.axis('equal')
    # 画外轮廓
    # plt.plot(out_contour[:, 0, 0], out_contour[:, 0, 1], color="black", linewidth=2)

    # 画左右连线
    y = (left_chest_leftmost[1] + right_chest_rightmost[1]) / 2
    xl = left_chest_leftmost[0]
    xr = right_chest_rightmost[0]

    plt.plot([xl, xr], [y, y], color="magenta", linewidth=2)

    x = bottom_sternum_point[0]
    yt = top_vertebra_point[1]
    yb = bottom_sternum_point[1]

    # 画e 
    plt.plot([x, x], [yt, yb], color="cyan", linewidth=2)


    plt.text(24, out_contour_top[1] - 24, "Width:%d, Hight:%d, Haller: %f." % (a, b, haller_index), fontsize=10, color="white")

    plt.legend()

    figure_image = fig2img(fig)

    return haller_index, figure_image