# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:13:56 2024

@author: Administrator
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

f = h5py.File('data/snr_real.h5','r')   
a = f["befor_snr"][:]
b = f["after_snr"][:]
f.close() 

# 绘制直方图
plt.figure(figsize=(8, 8))
bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 60)

plt.hist(a, bins=bins, alpha=0.5, label='SNR before', edgecolor='black',)
plt.hist(b, bins=bins, alpha=0.5, label='SNR after', edgecolor='black')

# 图例和标签
plt.xlabel('SNR', fontsize=24)
plt.ylabel('Frequency', fontsize=24)
#plt.title('SNR comparison', fontsize=26)
plt.legend(fontsize=24,frameon=False)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# 设置横纵坐标刻度字体大小
plt.xticks(fontsize=20)
plt.yticks(np.arange(0, 210, 50),fontsize=20)
plt.xlim(0, 19)

# 显示图形
plt.tight_layout()
plt.show()

f = h5py.File('data/cc_real.h5','r')   
a = f["cc"][:]
f.close() 
# 绘制直方图
plt.figure(figsize=(8, 8))
bins = np.linspace(a.min(), a.max(), 60)
plt.hist(a, bins=bins, alpha=0.5, label='CC', edgecolor='black',)

# 图例和标签
plt.xlabel('CC', fontsize=24)
plt.ylabel('Frequency', fontsize=24)
plt.title('CC', fontsize=26)
plt.legend(fontsize=24,frameon=False)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# 设置横纵坐标刻度字体大小
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0.9, 1.0)
# 显示图形
plt.tight_layout()
plt.show()



# 从文件中读取数据
f = h5py.File('data/cc_denoised_phase_diagram_normal.h5','r')   
a = f["real_cc"][:]
b = f["ideal_cc"][:]

#a = np.clip(a, 0.92, 1)
#b = np.clip(b, 0.92, 1)

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
plt.scatter(a, b, alpha=0.9, edgecolors='k', s=50)  # 原始散点
#plt.plot(a, b_pred, color='red', label=f'Fit: $y = {slope:.2f}x {intercept:.2f}$')  # 拟合直线
plt.plot(a, a, linestyle='--', color='gray')  # 添加 y=x 的虚线

# 设置图形属性
plt.title("Denoising Effect", fontsize=24)
plt.xlabel("Real CC", fontsize=24)
plt.ylabel("Ideal CC", fontsize=24)
plt.xlim(0.7, 1)
plt.ylim(0.7, 1)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20,frameon=False)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
