"""
生成标准的外轮廓，用于做轮廓比对，判断出ct是否为符合要求的横切图片
"""
import argparse
import cv2
import pydicom
import pickle
import glob
import os

from src.chest_diagnosis_v2 import find_outer_contour

parser = argparse.ArgumentParser()
parser.add_argument("src_dicom_folder", type=str, help="轮廓来源dicom文件")
parser.add_argument("output_file", type=str, help="轮廓输出文件夹")
args = parser.parse_args()

all_dicom_files = glob.glob(os.path.join(args.src_dicom_folder, "*"))
all_contours = []

for src_dicom in all_dicom_files:

    # 跳过非dicom格式的文件
    try:
        ds = pydicom.dcmread(src_dicom)
    except:
        continue

    img = cv2.convertScaleAbs(ds.pixel_array, alpha=(255.0/65535.0))

    # 提取像素轮廓点
    ret, binary = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 将所有轮廓按轮廓点数量由大到小排序
    contours = sorted(contours, key=lambda x: len(x))

    # 找到胸外轮廓(区域面积最大的为外胸廓轮廓点)
    out_contour, _ = find_outer_contour(contours)
    all_contours.append(out_contour)

with open(args.output_file, "wb") as f:
    pickle.dump(all_contours, f)
