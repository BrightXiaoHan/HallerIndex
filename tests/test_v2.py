import os
import numpy as np
from src import diagnosis_v2, depression_degree, is_avaliable

def test_chest_diagnosis_v2(dicom_test_files):

    for f in dicom_test_files:
        h, figure = diagnosis_v2(f)

def test_is_avaliable(dicom_test_files):
	for f in dicom_test_files:
		flag = is_avaliable(f)
		# assert flag is True

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

def test_depression_degree(filtered_patient):
	result = np.array([depression_degree(i) for i in filtered_patient])
	index = np.argmax(result)
	file_name = os.path.basename(filtered_patient[index])
	assert file_name == "FILE68"

