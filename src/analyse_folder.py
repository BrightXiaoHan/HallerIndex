import os
import numpy as np
from tradition.src import chest_diagnosis as tra_chest_diagnosis
from .utils import wrap_dicom_buffer, sort_files
from .chest_diagnosis import diagnosis, degree_of_depression, analyse, draw
from .contours_op import show_contours, show_points, find_boundary_point

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

    target_dic = None
    target_file = None
    for f, index in zip(sorted_files, indexes):
        f = wrap_dicom_buffer(f) if not isinstance(f, str) else f
        try:
            dic = analyse(f)
            tra_dic = tra_chest_diagnosis.analyse(f)
            dic.left_top = tra_dic.left_top
            dic.right_top = tra_dic.right_top
            dic.mid_bottom = tra_dic.mid_bottom
        except Exception as e:
            print(e)
            continue
        target_dic = dic
        target_file = f
        break        
    
    if target_dic is None:
        raise AvaliableDicomNotFoundException()
    
    haller_index, figure_image, a = draw(target_dic)
    
    if not _return_files:
        return [figure_image], [haller_index], a
    else:
        return [figure_image], [haller_index], [target_file], a

