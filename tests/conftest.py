import os
import sys
import glob
import pytest
import shutil

here = os.path.abspath(os.path.dirname(__file__))
source_root = os.path.abspath(os.path.join(here, '..'))

if source_root not in sys.path:
    sys.path.append(source_root)

@pytest.fixture(scope="session")
def dicom_test_files():
    """获取tests/assets中的所有待测试文件
    
    Returns:
        list: 文件路径列表
    """

    dicom_assets_folder = os.path.join(here, "assets/dicom_files")
    all_file = glob.glob(os.path.join(dicom_assets_folder, "*"))
    return all_file

@pytest.fixture(scope="session")
def filtered_patient():
    """获取手动过滤的病人ct影片
    
    Returns:
        list: 文件路径列表
    """
    filtered_patient_folder = os.path.join(source_root, "./data/filtered_patient")
    all_file = glob.glob(os.path.join(filtered_patient_folder, "FILE*"))
    return all_file

@pytest.fixture(scope="session")
def src_dest_mapping():
    """
    获取测试文件 -> 输出路径的位置映射

    Returns:
        dict: src -> dest mapping
    """
    mapping = dict()

    src_folder = os.path.join(source_root, "data/patient")
    target_folder = os.path.join(source_root, "data/result")

    patient_ids = os.listdir(src_folder)

    for pid in patient_ids:
        src_patient_folder = os.path.join(src_folder, pid)
        
        if not os.path.isdir(src_patient_folder):
            continue

        target_patient_folder = os.path.join(target_folder, pid)

        all_dicoms = os.listdir(src_patient_folder)


        for dicom_file in all_dicoms:
            mapping[os.path.join(src_patient_folder, dicom_file)] = os.path.join(target_patient_folder, dicom_file + ".png")
             
    return mapping
