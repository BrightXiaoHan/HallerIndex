"""
分析病人的所有ct数据，找出病人胸部最凹陷部位，计算Haller指数。
"""
import os
import glob
import pydicom
import cv2
import argparse

import numpy as np

from src import diagnosis_v2, depression_degree

parser = argparse.ArgumentParser()
parser.add_argument("src_dir", type=str, help="病人胸部ct文件夹路径。")
parser.add_argument("dest_dir", type=str, help="结果输出目录。")

args = parser.parse_args()

all_file = glob.glob(os.path.join(args.src_dir, "FILE*"))

degrees = np.array([depression_degree(i) for i in all_file])
index = np.argmax(degrees)
file_name = all_file[index]

haller_index, figure = diagnosis_v2(file_name)

figure.save(os.path.join(args.dest_dir, "result.png"))
with open(os.path.join(args.dest_dir, "haller.txt"), 'w') as f:
    f.write("Haller 指数值： %f" % haller_index)

    