from src import diagnosis_v2

def test_chest_diagnosis_v2(dicom_test_files):

    for f in dicom_test_files:
        h, figure = diagnosis_v2(f)


