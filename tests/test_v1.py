from src import diagnosis_v1

def test_chest_diagnosis_v1(dicom_test_files):

    for f in dicom_test_files:
        h, figure= diagnosis_v1(f)

