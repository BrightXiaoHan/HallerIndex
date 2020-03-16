# -*- coding: utf-8 -*-
"""
分析病人单张ct照片，计算Haller指数。
"""
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from src import wrap_dicom_buffer
from src import analyse, draw


def diagnosis_single(path, _min=False, _return_files=False):
    f = wrap_dicom_buffer(path)
    dic = analyse(f, _min)

    target_dic = dic
    target_file = f

    haller_index, figure_image, a, correction_index, asymmetry_index = draw(target_dic, path)

    if not _return_files:
        return [figure_image], [haller_index], a, correction_index, asymmetry_index
    else:
        return [figure_image], [haller_index], [target_file], a, correction_index, asymmetry_index


path = r'D:\lxy3\609964\FILE18'
dirname = os.path.dirname(path)
try:
    figures, indexes, fnames, a1, correction_index, asymmetry_index = diagnosis_single(path, _return_files=True, _min=False)
except:
    try:
        figures, indexes, fnames, a1, correction_index, asymmetry_index = diagnosis_single(path, _return_files=True, _min=True)
    except:
        print('ct照片不符合要求')

for i, (figure, name) in enumerate(zip(figures, fnames)):
    figure.save(os.path.join(dirname, "{}.png".format(os.path.basename(name))))
