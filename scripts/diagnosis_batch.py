"""
批量分析病人的所有ct数据，找出病人胸部最凹陷部位，计算Haller指数。
"""
import os
import glob
import pydicom
import cv2
import argparse
import shutil
import pylab
import numpy as np
from src import get_default_image
from src import diagnosis_folder
from tqdm import tqdm
from tradition.scripts.tra import tradition_func

parser = argparse.ArgumentParser()
parser.add_argument("src_dir", type=str, help="病人胸部ct文件夹路径。")
parser.add_argument("dest_dir", type=str, help="结果输出目录。")

args = parser.parse_args()

error_list = []
# 计算Haller指数，画辅助线
for folder in tqdm(list(os.walk(args.src_dir))[0][1]):
    if os.path.isdir(os.path.join(args.dest_dir, folder)):
        shutil.rmtree(os.path.join(args.dest_dir, folder))
    os.makedirs(os.path.join(args.dest_dir, folder))

    # path = os.path.join(os.path.join(args.src_dir, folder), "FILE*")
    # all_file = glob.glob(path)
    # for f in all_file:
    #     # 读取dicom文件中的像素数据
    #     try:
    #         ds = pydicom.dcmread(f)
    #     except pydicom.errors.InvalidDicomError as e:
    #         continue
    #
    #     saved_path = os.path.join(os.path.join(args.dest_dir, folder), os.path.basename(f) + ".png")
    #     try:
    #         pylab.imsave(saved_path, get_default_image(ds), cmap=pylab.cm.gray)
    #     except:
    #         continue

    try:
        figures, indexes, fnames, a1 = diagnosis_folder(os.path.join(args.src_dir, folder), _return_files=True, _debug=True)
        figures2, indexes2, fnames2, a2 = tradition_func(os.path.join(args.src_dir, folder))
        if abs(a2-a1) > 100:
            figures = figures2
            indexes = indexes2
            fnames = fnames2

    except Exception as e:
        error_list.append(folder)
        continue
    for i, (figure, name) in enumerate(zip(figures, fnames)):
        figure.save(os.path.join(args.dest_dir, folder, "{}_{}.png".format(folder, os.path.basename(name))))

for i in error_list:
    print("%s error." % i)

