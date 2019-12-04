import numpy as np

def get_pixels_hu(dc):
    """从dicom文件中获取hu值的图像
    
    Args:
        pixel_array (numpy.ndarray)): dicom.dcread的返回值
    
    Returns:
        numpy.ndarray: ct影像的hu值图像矩阵
    """
    # 转换为int16，int16是ok的，因为所有的数值都应该 <32k
    image = dc.pixel_array.astype(np.int16)

    # 设置边界外的元素为0
    image[image == -2000] = 0

    intercept = dc.RescaleIntercept
    slope = dc.RescaleSlope

    if slope != 1:
        image[slice_number] = slope * image.astype(np.float64)
        image[slice_number] = image.astype(np.int16)

    image += np.int16(intercept)

    return image

# 调整CT图像的窗宽窗位
def set_dicom_window_width_center(img_data, winwidth, wincenter, copy=True):
    """设置ct图像的窗宽，窗体中心值
    
    Args:
        img_data (numpy.ndarray): ct影像的hu值，get_pixels_hu的返回值
        winwidth (int): 窗宽
        wincenter (int): 窗中心值
        copy (bool, optional): 是否在原始矩阵的复制上面操作. Defaults to True.
    
    Returns:
        numpy.ndarray: 返回处理后的图像矩阵
    """
    if copy:
        img_temp = img_data.copy()
    else:
        img_temp = img_data
    img_temp.flags.writeable = True
    min_value = (2 * wincenter - winwidth) / 2.0 + 0.5
    max_value = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max_value - min_value)
    
    img_temp = (img_temp - min_value) * dFactor

    img_temp[img_temp < 0] = 0
    img_temp[img_temp > 255] = 255

    return img_temp