"""
批量分析病人的所有ct数据，找出病人胸部最凹陷部位，计算Haller指数。
"""
import os
import glob
import pydicom
import cv2
import argparse
import shutil

import numpy as np

from src import diagnosis_folder
from tqdm import tqdm

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
    try:
        figures, indexes, fnames = diagnosis_folder(os.path.join(args.src_dir, folder), _return_files=True)
    except Exception as e:
        error_list.append(folder)
        continue
    for i, (figure, name) in enumerate(zip(figures, fnames)):
        figure.save(os.path.join(args.dest_dir, folder, "%s.png" % os.path.basename(name)))

for i in error_list:
    print("%s error." % i)

