import os
import numpy as np
from .utils import wrap_dicom_buffer, sort_files
from .chest_diagnosis import degree_of_depression, analyse, draw
from .inner_degree import degree_of_inner


class AvaliableDicomNotFoundException(Exception):

    def __init__(self):
        super().__init__("没有找到符合条件的CT照片，请检查您上传的文件夹中是否有符合要求的横切照片。")


def diagnosis_folder_v2(folder, **kwargs):
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
    return diagnosis_files_v2(files, **kwargs)


def diagnosis_files_v2(files, _return_files=False, _debug=False, _min=False):
    """诊断一个病人所有的ct照片，
    注：默认文件是按顺序排列的 即 FILE1 FILE2 FILE3 ...

    Args:
        files (list): 列表的每个元素为文件路径或者二进制buffer
    Returns:
        list: PIL Image object list
        list: int list (Haller指数)
    """

    degrees = np.array([degree_of_depression(wrap_dicom_buffer(f)) for f in files])

    if degrees.max() <= 0:
        raise AvaliableDicomNotFoundException()

    # 找连续可用的照片
    start, end = 0, 0
    start_, end_ = 0, 0
    for i, (f, d) in enumerate(zip(files, degrees)):
        if d > 0:
            end += 1
        else:
            if end - start > end_ - start_:
                start_, end_ = start, end
            start = i
            end = i

    if end - start > end_ - start_:
        start_, end_ = start, end

    degrees = degrees[start_+1: end_]
    files = files[start_+1: end_]

    inner_degrees = np.array([degree_of_inner(wrap_dicom_buffer(f)) for f in files])
    max_index = np.argmax(inner_degrees)

    f = files[max_index]
    f = wrap_dicom_buffer(f)
    dic = analyse(f, _min)

    target_dic = dic
    target_file = f

    if target_dic is None:
        raise AvaliableDicomNotFoundException()

    haller_index, figure_image, a, correction_index, asymmetry_index = draw(target_dic, target_file)

    if not _return_files:
        return [figure_image], [haller_index], a, correction_index, asymmetry_index
    else:
        return [figure_image], [haller_index], [target_file], a, correction_index, asymmetry_index

