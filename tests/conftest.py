import os
import sys
import glob
import pytest

here = os.path.abspath(os.path.dirname(__file__))
source_root = os.path.abspath(os.path.join(here, '..'))

if source_root not in sys.path:
    sys.path.append(source_root)

@pytest.fixture(scope="session")
def dicom_test_files():
    dicom_assets_folder = os.path.join(here, "assets/dicom_files")
    all_file = glob.glob(os.path.join(dicom_assets_folder, "*"))
    return all_file


