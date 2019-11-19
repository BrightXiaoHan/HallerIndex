"""
从医院给的病例表格中提取出医生测量的Haller指数
"""
import argparse
import xlrd
import xlwt
import re
import os
from tqdm import tqdm

from src import diagnosis_folder, AvaliableDicomNotFoundException

parser = argparse.ArgumentParser()
parser.add_argument("input_excel", type=str, help="输入表格")
parser.add_argument("is_processed", type=bool, help="表示表格是否已经处理过了，True表示已经经过处理，表格只有两列，第一列是住院号，第二列是Haller指数。没有处理过的是原始表格，里面包含诊断意见，需要使用正则表达式抽取Haller指数。")
parser.add_argument("output_excel", type=str, help="输出表格")
parser.add_argument("decom_dir", type=str, help="ct影像目录，目录中根据病人住院号命名的文件夹，每个文件夹中是病人的ct照片")

args = parser.parse_args()

# 提取Haller指数的正则表达式
haller_reg = r".*(Haller.{1,5})([0-9]\.[0-9]{1,2}).*"


# 打开工作表
workbook = xlrd.open_workbook(filename=args.input_excel)
# 用索引取第一个工作薄
booksheet = workbook.sheet_by_index(0)

mapping = {}
# 返回的结果集
for i in range(1, booksheet.nrows):
    value = booksheet.row_values(i)
    pid = str(int(value[0]))
    description = value[-1]
    if args.is_processed:
        haller_index = float(description)
    else:
        match_obj = re.match(haller_reg, description)
        if not match_obj:
            continue
        haller_index = match_obj.group(2)
    mapping[pid] = {"ground_truth": haller_index, "prediction": -1}


# 从目标文件夹中寻找结果
for folder_name in tqdm(os.listdir(args.decom_dir)):
    folder = os.path.join(args.decom_dir, folder_name)
    if os.path.isdir(folder) and folder_name in mapping:
        try:
            _, h = diagnosis_folder(folder)
        except AvaliableDicomNotFoundException:
            continue
        if len(h) > 0:
            mapping[folder_name]["prediction"] = max(h)
        

workbook = xlwt.Workbook(encoding='utf-8')
booksheet = workbook.add_sheet('Sheet1', cell_overwrite_ok=True)

# 写列头
row = 0
header = ["住院号", "医院测量Haller指数", "计算机测量Haller指数"]
for col in range(len(header)):
    booksheet.write(row, col, header[col])

# 写内容
for key, value in mapping.items():
    row += 1
    booksheet.write(row, 0, key)
    booksheet.write(row, 1, value["ground_truth"])
    booksheet.write(row, 2, value["prediction"])

# 保存文件
workbook.save(args.output_excel)

    

    
