"""将一个病人的所有dicom文件转化为jpg文件
"""
import os
import glob
import pydicom
import cv2
import argparse
import pylab

parser = argparse.ArgumentParser()
parser.add_argument("src_dir", type=str, help="病人胸部ct文件夹路径。")
parser.add_argument("dest_dir", type=str, help="结果输出目录。")

args = parser.parse_args()

all_file = glob.glob(os.path.join(args.src_dir, "FILE*"))

if not os.path.isdir(args.dest_dir):
     os.makedirs(args.dest_dir)

for f in all_file:
	# 读取dicom文件中的像素数据
    try:
        ds = pydicom.dcmread(f)
    except pydicom.errors.InvalidDicomError as e:
        continue

    saved_path = os.path.join(args.dest_dir, os.path.basename(f) + ".jpg")
    pylab.imsave(saved_path, ds.pixel_array, cmap=pylab.cm.gray)
