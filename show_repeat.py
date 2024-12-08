# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:13:56 2024

@author: Administrator
"""

import h5py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

f = h5py.File('data/snr_repeat.h5','r')   
a = f["befor_snr"][:]
b = f["after_snr"][:]
f.close() 

# 绘制直方图
plt.figure(figsize=(10, 6))
bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 30)

plt.hist(a, bins=bins, alpha=0.5, label='SNR before', edgecolor='black',)
plt.hist(b, bins=bins, alpha=0.5, label='SNR after', edgecolor='black')

# 图例和标签
plt.xlabel('SNR', fontsize=24)
plt.ylabel('Frequency', fontsize=24)
plt.title('SNR comparison', fontsize=26)
plt.legend(fontsize=24,frameon=False)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# 设置横纵坐标刻度字体大小
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, 20)  # 设置横坐标范围

# 读取数据
data_files = [
    ('data/cc_repeat_all_befor.h5', "CC(x,y)"),
    ('data/cc_repeat_1after_2before.h5', "CC(x,y')"),
]

# 绘制直方图
plt.figure(figsize=(12, 8))
for file_path, label in data_files:
    with h5py.File(file_path, 'r') as f:
        a = f["cc"][:]
    bins = np.arange(0.2, 1.0 + 0.025, 0.05)
    plt.hist(a, bins=bins, alpha=0.5, label=label, edgecolor='black')

# 图例和标签
plt.xlabel("CC", fontsize=24)
plt.ylabel('Frequency', fontsize=24)
plt.legend(fontsize=32, loc='upper left', frameon=False)  # 左上角，移除框线
plt.grid(axis='y', linestyle='--', alpha=0.7)
# 设置横纵坐标刻度字体大小
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.xlim(0.2, 1)  # 限制横坐标范围
# 显示图形
plt.tight_layout()
plt.show()

# 读取数据
data_files = [
    ('data/cc_repeat_all_befor.h5', "CC(x,y)"),
    ('data/cc_repeat_1beforer_2after.h5', "CC(x',y)")
]

# 绘制直方图
plt.figure(figsize=(12, 8))
for file_path, label in data_files:
    with h5py.File(file_path, 'r') as f:
        a = f["cc"][:]
    bins = np.arange(0.2, 1.0 + 0.025, 0.05)
    plt.hist(a, bins=bins, alpha=0.5, label=label, edgecolor='black')

# 图例和标签
plt.xlabel("CC", fontsize=24)
plt.ylabel('Frequency', fontsize=24)
plt.legend(fontsize=32, loc='upper left', frameon=False)  # 左上角，移除框线
plt.grid(axis='y', linestyle='--', alpha=0.7)
# 设置横纵坐标刻度字体大小
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.xlim(0.2, 1)  # 限制横坐标范围
# 显示图形
plt.tight_layout()
plt.show()

# 读取数据
data_files = [
    ('data/cc_repeat_all_befor.h5', "CC(x,y)"),
    ('data/cc_repeat_all_after.h5', "CC(x',y')"),
]

# 绘制直方图
plt.figure(figsize=(12, 8))
for file_path, label in data_files:
    with h5py.File(file_path, 'r') as f:
        a = f["cc"][:]
    bins = np.arange(0.2, 1.0 + 0.025, 0.05)
    plt.hist(a, bins=bins, alpha=0.5, label=label, edgecolor='black')

# 图例和标签
plt.xlabel("CC", fontsize=24)
plt.ylabel('Frequency', fontsize=24)
plt.legend(fontsize=32, loc='upper left', frameon=False)  # 左上角，移除框线
plt.grid(axis='y', linestyle='--', alpha=0.7)
# 设置横纵坐标刻度字体大小
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.xlim(0.2, 1)  # 限制横坐标范围
# 显示图形
plt.tight_layout()
plt.show()



# 从文件中读取数据
with h5py.File('data/cc_denoised_phase_diagram.h5', 'r') as f:
    a = f["real_cc"][:]
    b = f["ideal_cc"][:]
    
# 计算残差
residuals = b - a

# 绘制残差的直方图
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=100, color='blue', alpha=0.7, edgecolor='black')
plt.axvline(0, color='red', linestyle='--', label='Zero Residual')
plt.title("Histogram of Residuals", fontsize=20)
plt.xlabel("Residuals (Ideal - Real)", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14,frameon=False)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(-0.2, 0.2)  # 设置横坐标范围
plt.ylim(0, 600)  # 设置横坐标范围
plt.show()
    
# 将 a 作为输入，b 作为输出
a = a.reshape(-1, 1)  # 转换为列向量
model = LinearRegression()
model.fit(a, b)  # 使用 a 作为输入，b 作为输出

# 获取拟合参数
slope = model.coef_[0]  # 斜率 m
intercept = model.intercept_  # 截距 c

# 计算预测值
b_pred = model.predict(a)

# 计算 R^2 值
r2 = r2_score(b, b_pred)
print(f"线性拟合模型: b = {slope:.4f} * a + {intercept:.4f}")
print(f"拟合优度 R^2: {r2:.4f}")

# 绘制图形
plt.figure(figsize=(8, 8))
plt.scatter(a, b, alpha=0.9, edgecolors='k', s=50, label="Data")  # 原始散点
plt.plot(a, b_pred, color='red', label=f'Fit: $y = {slope:.2f}x {intercept:.2f}$')  # 拟合直线
plt.plot(a, a, linestyle='--', color='gray', label='y=x')  # 添加 y=x 的虚线
# 设置图形属性
plt.title("Denoised Effect", fontsize=24)
plt.xlabel("Real CC", fontsize=24)
plt.ylabel("Ideal CC", fontsize=24)
plt.xlim(0.7, 1)
plt.ylim(0.7, 1)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20,frameon=False)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()




# 从文件中读取数据
with h5py.File('data/cc_repeat_all_befor.h5', 'r') as f:
    a = f["cc"][:]
with h5py.File('data/cc_repeat_all_after.h5', 'r') as f:
    b = f["cc"][:]



plt.figure(figsize=(8, 8))
plt.scatter(a, b, alpha=0.7, edgecolors='k', s=50)
# 添加 y = x 的虚线
x = np.linspace(0.5, 1, 100)  # 定义 x 的范围
plt.plot(x, x, linestyle='--', color='gray', label='y=x')  # 添加虚线
plt.title("CC before and after denoising", fontsize=24)
plt.xlabel("Before denoising", fontsize=20)
plt.ylabel("Afetr denoising", fontsize=20)
plt.xlim(0.5, 1)  # 限制横坐标范围
plt.ylim(0.5, 1)  # 限制纵坐标范围
# 调整横纵坐标刻度的字号
plt.xticks(fontsize=20)  # 横坐标刻度字号
plt.yticks(fontsize=20)  # 纵坐标刻度字号
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()