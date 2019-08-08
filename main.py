import os
import traceback
import shutil
from chest_diagnosis import diagnosis

if __name__ == "__main__":

    if os.path.isdir("result"):
        shutil.rmtree("result")

    for patient in os.listdir("patient"):
        dir_path = os.path.join("patient", patient)
        all_file = os.listdir(dir_path)
        for f in all_file:
            input_file = os.path.join("patient", patient, f)
            output_file = os.path.join("result", patient, f + ".jpg")
            if not os.path.isdir(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            try:
                h1, h2 = diagnosis(input_file, output_file)
            except Exception as e:
                traceback.print_stack()
                traceback.
                print(e)
                continue