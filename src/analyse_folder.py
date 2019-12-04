import os
import numpy as np

from .utils import wrap_dicom_buffer, sort_files
from .chest_diagnosis import diagnosis, degree_of_depression, analyse, draw
from .contours_op import show_contours, show_points

class AvaliableDicomNotFoundException(Exception):
    
    def __init__(self):
        super().__init__("没有找到符合条件的CT照片，请检查您上传的文件夹中是否有符合要求的横切照片。")

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
    files = sort_files(files)
    return diagnosis_files(files, **kwargs)


def diagnosis_files(files, _return_files=False, _debug=False):
    """诊断一个病人所有的ct照片，
    注：默认文件是按顺序排列的 即 FILE1 FILE2 FILE3 ...
    
    Args:
        files (list): 列表的每个元素为文件路径或者二进制buffer
    Returns:
        list: PIL Image object list
        list: int list (Haller指数)
    """
    degrees = []

    degrees = np.array([degree_of_depression(wrap_dicom_buffer(f)) for f in files])
    
    if degrees.max() <= 0:
        raise AvaliableDicomNotFoundException()

    # 找连续可用的照片
    start, end = 0, 0
    start_, end_ = 0, 0
    for i ,(f, d) in enumerate(zip(files, degrees)):
        if d > 0:
            end +=1
        else:
            if end - start > end_ - start_:
                start_, end_ = start, end
            start = i
            end = i

    if end - start > end_ - start_:
        start_, end_ = start, end
    
    degrees = degrees[start_: end_]
    files = files[start_: end_]

    indexes = np.argsort(degrees)
    
    sorted_files = [files[i] for i in indexes]
    sorted_files.reverse()
    indexes = indexes.tolist()
    indexes.reverse()

    target_img = None
    target_dic = None
    target_file = None
    target_index = None
    for f, index in zip(sorted_files, indexes):
        f = wrap_dicom_buffer(f) if not isinstance(f, str) else f
        try:
            dic = analyse(f)
        except:
            continue
        target_dic = dic
        target_file = sorted_files[index]
        target_img = dic.img
        target_index = index    
        break        
    
    if target_index is None:
        raise AvaliableDicomNotFoundException()
    
    # 找到凹陷程度最大的照片和它的临近照片
    max_index = target_index
    a = max_index - 2 if max_index - 2 > 0 else 0
    b = max_index + 2 if max_index + 2 < len(files) else len(files)
    neibor_files = [files[i] for i in range(a, b) if i != max_index]

    # 分析这张照片，找出关键点和关键轮廓
    countours_set = []
    points_set = []
    for f in neibor_files:
        f = wrap_dicom_buffer(f) if not isinstance(f, str) else f
        try:
            dic = analyse(f)
            if _debug:
                countours_set.append(dic.vertebra)
        except:
            continue
    countours_set.append(target_dic.vertebra)
    
    show_contours(target_img, countours_set)
    haller_index, figure_image = draw(target_dic)
    
    if not _return_files:
        return [figure_image], [haller_index]
    else:
        return [figure_image], [haller_index], [target_file]

