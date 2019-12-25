# 将内轮廓图片的边缘,映射到原图
import cv2

def fusion(masks, original_images):
    cnt_list = []
    for mask, origin_image in zip(masks, original_images):

        size = origin_image.shape

        # 将轮廓图放大至与原图相等
        image_resize = cv2.resize(mask, (size[1], size[0]))
        # 二值化
        ret, binary = cv2.threshold(image_resize, 180, 255, cv2.THRESH_BINARY)
        # 找轮廓
        img, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 图像宽和高
        h, w = binary.shape
        # 最大轮廓面积不能超过这个
        min_cnt_area = h*w-10000

        # 轮廓坐标
        min_cnt = None
        max_area = 0
        # 找出目标轮廓
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_cnt_area:
                if area > max_area:
                    max_area = area
                    min_cnt = cnt

        cnt_list.append(min_cnt)

    return cnt_list

