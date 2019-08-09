# %%
# 导入依赖包
import os
import re
import pydicom
import cv2

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.optimize import leastsq
from scipy.interpolate import splev, splprep

# %%
# 读取dicom文件
folder = "data/pe"
f = os.listdir(folder)[6]
print(f)
ds = pydicom.dcmread(os.path.join(folder, f))  # plan dataset
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
img = cv2.convertScaleAbs(ds.pixel_array, alpha=(255.0/65535.0))
plt.imshow(img, cmap=plt.cm.bone)


# %%
# 提取像素轮廓点
ret, binary = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)
cim2, contours, hierarchy = cv2.findContours(
    binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = max(contours, key=lambda x: len(x))
contours = sorted(contours, key=lambda x: len(x))

img_with_contours = np.copy(img)
cv2.drawContours(img_with_contours, contours[-1:], -1, (255, 255, 255), 3)
plt.imshow(img_with_contours, cmap=plt.cm.bone)

# %%
# PCA主成分分析
pca = PCA(n_components=2)
pca_contours = np.expand_dims(
    pca.fit_transform(np.squeeze(contours[-1])), axis=1)

x = pca_contours[:, 0, 0]
y = -pca_contours[:, 0, 1]


# %%
# 根据样本点拟合椭圆曲线
def error(p, x, y):
    return np.sqrt((p[1])**2 - ((x + p[2])**2)*(p[1]**2) / (p[0]**2)) - np.abs(y+p[3])

p0 = [max(x) - min(x), max(y) - min(y), 0, 0]
ret = leastsq(error, p0, args=(x, y))
a, b, offset_x, offset_y = ret[0]

# 矫正点坐标
x += offset_x
y += offset_y

# 画出拟合曲线和原始点集
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color="red", label="Sample Point", linewidth=3)
fit_x = np.linspace(-a, a, 1000)
fit_y = np.sqrt(b**2 - (fit_x)**2*b**2/a**2)
plt.plot(fit_x, fit_y, color="orange", label="Fitting Curve", linewidth=2)
plt.plot(fit_x, -fit_y, color="orange", label="Fitting Curve", linewidth=2)
plt.legend()
plt.show()

# %%
# 将采样的样本点根据极坐标排序
def sort_clockwise(x, y):
    """
    sort coordinates clockwise

    Args:
        x(np.ndarray): with shape (n)
        y(np.ndarray): with shape (n)
    Return:
        np.ndarray: with shape (n, 1, 2)
    """

    plural = x + y * 1j

    angle = np.angle(plural)

    sort_keys = np.argsort(angle)

    return x[sort_keys], y[sort_keys]

x, y = sort_clockwise(x, y)

tck, u = splprep([x[y >= 0], y[y >= 0]], s = 0)
flatten_x_pos, flatten_y_pos = splev(u,tck)

tck, u = splprep([x[y < 0], y[y < 0]], s = 0)
flatten_x_neg, flatten_y_neg = splev(u,tck)

plt.plot(flatten_x_pos, flatten_y_pos, 'r-', flatten_x_neg, flatten_y_neg, 'r-')
plt.legend(["Fit line."],loc='best')
plt.show()

#%%
# 如果是凹型的，取最凹点
mid_index = np.argmin(np.abs(flatten_x_pos))  # 获取中间点
left_xs, left_ys = flatten_x_pos[mid_index:], flatten_y_pos[mid_index:]
right_xs, right_ys = flatten_x_pos[:
                                    mid_index], flatten_y_pos[:mid_index]

max_left_index = np.argmax(left_ys)  # 计算左侧凸点
max_left_x, max_left_y = left_xs[max_left_index], left_ys[max_left_index]

max_right_index = np.argmax(right_ys)  # 计算右侧凸点
max_right_x, max_right_y = right_xs[max_right_index], right_ys[max_right_index]

mid_xs, mid_ys = flatten_x_pos[max_right_index: max_left_index +
                                mid_index], flatten_y_pos[max_right_index: max_left_index + mid_index]
min_index = np.argmin(mid_ys)
min_x, min_y = mid_xs[min_index], mid_ys[min_index]  # 计算中间凹点

A = np.array([min_x, min_y])

B = np.array([A[0], 0])

C = np.array([-a, 0])

_E = np.array([0, A[1]])

d = 2 * a

e = A[0]

# 计算H1，H2分型指数
H1 = d / abs(A[1])
H2 = 2 * e / d * np.sign(abs(B[0] - C[0]) - d/2)

#%%
plt.figure(figsize=(8, 6))
# 画胸廓拟合点集
plt.plot(np.concatenate([flatten_x_pos, flatten_x_neg]), np.concatenate(
    [flatten_y_pos, flatten_y_neg]), color="orange", label="Fitted Curve", linewidth=2)

# 画椭圆以及椭圆内径
oval_long_axis_x = np.linspace(-a, a, 1000)
oval_long_axis_y = np.zeros((1000,))
oval_short_axis_x = np.zeros((500,))
oval_short_axis_y = np.linspace(-b, b, 500)
# 画椭圆
plt.plot(np.concatenate([fit_x, fit_x]), np.concatenate(
    [fit_y, -fit_y]), color="blue", label="Fitted oval", linewidth=2)
# 画长轴
plt.plot(oval_long_axis_x, oval_long_axis_y, color="yellow", label="d=%d pixels" % (2*a), linewidth=2)
# 画短轴
plt.plot(oval_short_axis_x, oval_short_axis_y,
            color="blue", linewidth=2)

# 画A， B
plt.plot(*zip(*[A, B]), color="magenta", label="AB=%d pixels" % (A[1] - B[1]).astype(int), linewidth=4)

# 画e 
plt.plot(*zip(*[A, _E]), color="cyan", label="e=%d pixels" % (np.abs(A[0] - _E[0])).astype(int), linewidth=4)

plt.text(*A, "A", fontsize=24)
plt.text(*B, "B", fontsize=24)
plt.text(*C, "C", fontsize=24)

plt.text(0, -24, "H1: %f, H2: %f" % (H1, H2), fontsize=24)

plt.legend()
plt.show()

#%%
