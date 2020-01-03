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

from tradition.src import diagnosis_folder
from tqdm import tqdm


def tradition_func(path):
    figures, indexes, fnames, a = diagnosis_folder(path, _return_files=True, _debug=True)
    return figures, indexes, fnames, a



