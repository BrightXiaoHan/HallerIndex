from .chest_diagnosis_v2 import diagnosis as diagnosis_v2
from .chest_diagnosis_v2 import depression_degree, is_avaliable

import os
import numpy as np
from .utils import wrap_dicom_buffer

def diagnosis_folder(folder, top=3):
    """诊断一个病人所有的ct照片
    
    Args:
        folder (list): 文件夹路径。文件夹包含一个病人的所有ct照片
        top (int): default is 3. 选择Haller指数最大的 top 张照片作为返回值
    Returns:
        list: PIL Image object list
        list: int list (Haller指数)
    """
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    return diagnosis_files(files)


def diagnosis_files(files, top=3):
    """诊断一个病人所有的ct照片
    
    Args:
        files (list): 列表的每个元素为文件路径或者二进制buffer
        top (int): default is 3. 选择Haller指数最大的 top 张照片作为返回值
    Returns:
        list: PIL Image object list
        list: int list (Haller指数)
    """
    degrees = []
    avaliable_files = []

    for f in files:
        try:
            f = wrap_dicom_buffer(f) if isinstance(f, bytes) else f 
            degrees.append(diagnosis_v2(f, plot=False)[0])
        except Exception as e:
            # print(e)
            continue
        avaliable_files.append(f)

    degrees = np.array(degrees)
    indexes = np.argsort(degrees)
    if len(indexes) >= top:
        indexes = indexes[-top:]
    
    files = [avaliable_files[i] for i in indexes]
    files.reverse()

    figure_set = []
    haller_set = []
    for f in files:
        try:
            f = wrap_dicom_buffer(f) if not isinstance(f, str) else f
            haller, figure = diagnosis_v2(f)
        except Exception as e:
            # print(e)
            continue
        figure_set.append(figure)
        haller_set.append(haller)

    return figure_set, haller_set


__all__ = ["diagnosis_v2", "depression_degree", "is_avaliable", "diagnosis_files"]