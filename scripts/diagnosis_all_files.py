"""
分析病人的所有ct数据，保存每一张诊断图片
"""
import os
import glob
import pydicom
import cv2
import argparse

import numpy as np

from src import diagnosis_folder

parser = argparse.ArgumentParser()
parser.add_argument("src_dir", type=str, help="病人胸部ct文件夹路径。")
parser.add_argument("dest_dir", type=str, help="结果输出目录。")

args = parser.parse_args()

# 计算Haller指数，画辅助线
figure, haller_index, files = diagnosis_folder(args.src_dir, top=-1,  _return_files=True)

if not os.path.isdir(args.dest_dir):
    os.makedirs(args.dest_dir)

for img, f in zip(figure, files):
    output_path = os.path.join(args.dest_dir, os.path.basename(f)) + ".png"
    img.save(output_path)
