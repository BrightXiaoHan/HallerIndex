import os
import numpy as np

from .utils import wrap_dicom_buffer
from .chest_diagnosis import diagnosis, degree_of_depression

def diagnosis_folder(folder, **kwargs):
    """诊断一个病人所有的ct照片
    
    Args:
        folder (list): 文件夹路径。文件夹包含一个病人的所有ct照片
        top (int): default is 3. 选择Haller指数最大的 top 张照片作为返回值
    Returns:
        list: PIL Image object list
        list: int list (Haller指数)
    """
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    return diagnosis_files(files, **kwargs)


def diagnosis_files(files, top=3, _return_files=False):
    """诊断一个病人所有的ct照片
    
    Args:
        files (list): 列表的每个元素为文件路径或者二进制buffer
        top (int): default is 3. 选择Haller指数最大的 top 张照片作为返回值
    Returns:
        list: PIL Image object list
        list: int list (Haller指数)
    """
    degrees = []

    degrees = np.array([degree_of_depression(f) for f in files])
    indexes = np.argsort(degrees)
    if top > 0:
        if len(indexes) >= top:
            indexes = indexes[-top:]
    
    files = [files[i] for i in indexes]
    files.reverse()

    figure_set = []
    haller_set = []
    files_set = []
    for f in files:
        try:
            f = wrap_dicom_buffer(f) if not isinstance(f, str) else f
            haller, figure = diagnosis(f)
        except Exception as e:
            continue
        figure_set.append(figure)
        haller_set.append(haller)
        files_set.append(f)

    if not _return_files:
        return figure_set, haller_set
    else:
        return figure_set, haller_set, files_set


__all__ = ["diagnosis_v2", "depression_degree", "is_avaliable", "diagnosis_files"]