import os
from src import diagnosis_v2

def test_chest_diagnosis_v2(dicom_test_files):

    for f in dicom_test_files:
        h, figure = diagnosis_v2(f)


def test_batch(src_dest_mapping):
	
	error_list = []

	for src, target in src_dest_mapping.items():
		try:
			h, figure = diagnosis_v2(src)
		except:
			error_list.append(src)
			continue

		if not os.path.isdir(os.path.dirname(target)):
			os.makedirs(os.path.dirname(target))

		figure.save(target)
