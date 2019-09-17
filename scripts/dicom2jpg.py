"""将一个病人的所有dicom文件转化为jpg文件
"""
import os
import glob
import pydicom
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("src_dir", type=str, required=True, help="病人胸部ct文件夹路径。")
parser.add_argument("dest_dir", type=str, required=True, help="结果输出目录。")

args = parser.parse_args()

all_file = glob.glob(os.path.join(args.src_dir, "FILE*"))

for i, f in enumerate(all_file):
     # 读取dicom文件中的像素数据
    ds = pydicom.dcmread(f)
    img = cv2.convertScaleAbs(ds.pixel_array, alpha=(255.0/65535.0))
    cv2.imwrite(os.path.join(args.dest_dir, str(i) + ".jpg"), img * 20)
