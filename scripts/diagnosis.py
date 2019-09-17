"""
分析病人的所有ct数据，找出病人胸部最凹陷部位，计算Haller指数。
"""
import os
import glob
import pydicom
import cv2
import argparse

from src import diagnosis_v2

parser = argparse.ArgumentParser()
parser.add_argument("src_dir", type=str, required=True, help="病人胸部ct文件夹路径。")
parser.add_argument("dest_dir", type=str, required=True, help="结果输出目录。")

args = parser.parse_args()

all_file = glob.glob(os.path.join(args.src_dir, "FILE*"))

for i, f in enumerate(all_file):
    pass