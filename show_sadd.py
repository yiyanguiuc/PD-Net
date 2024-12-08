# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:34:21 2024

@author: Administrator
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from scipy import signal
import matplotlib.pyplot as plt
from Utils import *
import numpy as np
import h5py
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Arial'

output_nz = 2801

f = h5py.File('data/repeat_test.h5','r')   
test_input1 = f["data1"][:]
test_input2 = f["data2"][:]
arc = f["arc"][:]
f.close()

n_test = test_input1.shape[0]



f = h5py.File('data/saad_result.h5','r')   
data1 = f["data1"][:]
data2 = f["data2"][:]
f.close() 



numberr = 8
x=np.arange(0,2400)
rf1=test_input1[numberr,200:2600]
rf2=test_input1[numberr,200:2600] - data1[numberr,200:2600]
rf1 = np.reshape(rf1,(len(rf1)))
rf2 = np.reshape(rf2,(len(rf2)))
#plt.plot(x,rf1,label="Repeat1 before denoised")
plt.plot(x,rf2,label="noise")
#print("Denoised.max",outs3[numberr,:].max())
plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.,fontsize=18, frameon=False)
plt.show()




outs1 = test_input1[:,:]
outs3 = data1[:,:]
outs2 = test_input2[:,:]
outs4 = data2[:,:]

def cc_with_offset(x, y, beg, end):
    # Ensure x and y are tensors
    if isinstance(x, np.ndarray):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
    if isinstance(y, np.ndarray):
        y = tf.convert_to_tensor(y, dtype=tf.float32)
    
    # Use sliding window to extract patches (equivalent to PyTorch unfold)
    yy_slices = tf.image.extract_patches(
        images=tf.expand_dims(tf.expand_dims(y, axis=1), axis=-1),  # Add batch and channel dimensions
        sizes=[1, 1, end - beg, 1],  # Patch size
        strides=[1, 1, 1, 1],        # Strides
        rates=[1, 1, 1, 1],          # Dilation rate
        padding='VALID'              # No padding
    )

    # Remove extra dimensions and transpose for compatibility
    yy_slices = tf.squeeze(yy_slices, axis=1)  # Shape: (batch_size, num_patches, patch_size)
    yy_slices = tf.transpose(yy_slices, perm=[0, 2, 1])  # Shape: (batch_size, patch_size, num_patches)
    yy_slices = yy_slices[:, :, beg - 200:beg + 200]  # Adjust index range

    # Fixed time window for x
    x1 = x[:, beg:end]  # Shape: (batch_size, end - beg)

    # Mean calculation
    mean_x1 = tf.reduce_mean(x1, axis=1, keepdims=True)[:, :, tf.newaxis]  # Shape: (batch_size, 1, 1)
    mean_yy_slices = tf.reduce_mean(yy_slices, axis=1, keepdims=True)      # Shape: (batch_size, 1, num_patches)

    # Covariance calculation
    covariance_yy = tf.reduce_sum((x1[:, :, tf.newaxis] - mean_x1) * (yy_slices - mean_yy_slices), axis=1)

    # Standard deviation calculation
    std_x1 = tf.sqrt(tf.reduce_sum((x1[:, :, tf.newaxis] - mean_x1) ** 2, axis=1))
    std_yy_slices = tf.sqrt(tf.reduce_sum((yy_slices - mean_yy_slices) ** 2, axis=1))

    # Correlation coefficient calculation
    correlation_yy = covariance_yy / (std_x1 * std_yy_slices + 1e-8)

    # Maximum correlation and indices
    max_correlation_yy = tf.reduce_max(correlation_yy, axis=1)
    max_indices = tf.argmax(correlation_yy, axis=1)

    return tf.reduce_mean(max_correlation_yy), max_indices

def cc(x, y, beg, end):
    # Ensure x and y are tensors
    if isinstance(x, np.ndarray):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
    if isinstance(y, np.ndarray):
        y = tf.convert_to_tensor(y, dtype=tf.float32)
    
    # Use sliding window to extract patches (equivalent to PyTorch unfold)
    yy_slices = tf.image.extract_patches(
        images=tf.expand_dims(tf.expand_dims(y, axis=1), axis=-1),  # Add batch and channel dimensions
        sizes=[1, 1, end - beg, 1],  # Patch size
        strides=[1, 1, 1, 1],        # Strides
        rates=[1, 1, 1, 1],          # Dilation rate
        padding='VALID'              # No padding
    )

    # Remove extra dimensions and transpose for compatibility
    yy_slices = tf.squeeze(yy_slices, axis=1)  # Shape: (batch_size, num_patches, patch_size)
    yy_slices = tf.transpose(yy_slices, perm=[0, 2, 1])  # Shape: (batch_size, patch_size, num_patches)
    yy_slices = yy_slices[:, :, beg - 200:beg + 200]  # Adjust index range

    # Fixed time window for x
    x1 = x[:, beg:end]  # Shape: (batch_size, end - beg)

    # Mean calculation
    mean_x1 = tf.reduce_mean(x1, axis=1, keepdims=True)[:, :, tf.newaxis]  # Shape: (batch_size, 1, 1)
    mean_yy_slices = tf.reduce_mean(yy_slices, axis=1, keepdims=True)      # Shape: (batch_size, 1, num_patches)

    # Covariance calculation
    covariance_yy = tf.reduce_sum((x1[:, :, tf.newaxis] - mean_x1) * (yy_slices - mean_yy_slices), axis=1)

    # Standard deviation calculation
    std_x1 = tf.sqrt(tf.reduce_sum((x1[:, :, tf.newaxis] - mean_x1) ** 2, axis=1))
    std_yy_slices = tf.sqrt(tf.reduce_sum((yy_slices - mean_yy_slices) ** 2, axis=1))

    # Correlation coefficient calculation
    correlation_yy = covariance_yy / (std_x1 * std_yy_slices + 1e-8)

    # Maximum correlation and indices
    max_correlation_yy = tf.reduce_max(correlation_yy, axis=1)
    max_indices = tf.argmax(correlation_yy, axis=1)


    return max_correlation_yy, max_indices


corr_before, i1 = cc_with_offset(outs1,outs2, 1600, 2600)
print("1,2 Before denoising corrlation=",corr_before.numpy())
corr_after, i2 = cc_with_offset(outs3,outs4, 1600, 2600)
print("1,2 after denoising corrlation=",corr_after.numpy())
corr, i3 = cc_with_offset(outs1,outs4, 1600, 2600)
print("1 before and 2 after corrlation=",corr.numpy())
corr, i4 = cc_with_offset(outs2,outs3, 1600, 2600)
print("1 after and 2 before corrlation=",corr.numpy())


corr,offet = cc(outs1,outs2, 1600, 2600)
corr_before = corr.numpy()
corr,offet = cc(outs3,outs4, 1600, 2600)
corr_after = corr.numpy()


i1_snr = 0
i2_snr = 0
before_snr = []
after_snr = []


for number in range(n_test):
    noise1 = 0
    noise2 = 0
    noise3 = 0
    noise4 = 0
    for j in range(0,1400):
        noise1 = noise1 + outs1[number,j] * outs1[number,j]
        noise2 = noise2 + outs2[number,j] * outs2[number,j]
        noise3 = noise3 + outs3[number,j] * outs3[number,j]
        noise4 = noise4 + outs4[number,j] * outs4[number,j]
    noise1 = noise1 / 1400
    noise2 = noise2 / 1400
    noise3 = noise3 / 1400
    noise4 = noise4 / 1400
    max1 = 0
    max2 = 0
    max3 = 0
    max4 = 0
    for j in range(1600,2600):
        max1 = max1 + outs1[number,j] * outs1[number,j]
        max2 = max2 + outs2[number,j] * outs2[number,j]
        max3 = max3 + outs3[number,j] * outs3[number,j]
        max4 = max4 + outs4[number,j] * outs4[number,j]
        
    max1 = max1 / 1000
    max2 = max2 / 1000
    max3 = max3 / 1000
    max4 = max4 / 1000

    i1_snr = i1_snr + 10*np.log10(math.sqrt(max1/noise1)) + 10*np.log10(math.sqrt(max2/noise2))
    i2_snr = i2_snr + 10*np.log10(math.sqrt(max3/noise3)) + 10*np.log10(math.sqrt(max4/noise4))
    before_snr.append((10*np.log10(math.sqrt(max1/noise1)) + 10*np.log10(math.sqrt(max2/noise2)))/2)
    after_snr.append((10*np.log10(math.sqrt(max3/noise3)) + 10*np.log10(math.sqrt(max4/noise4)))/2)
print("Repeat earthquakes Before denoising SNR=",i1_snr/n_test/2)
print("After denoising SNR=",i2_snr/n_test/2)


ideal_cc = []
real_cc = []
ideal_cc_avg1 = 0
ideal_cc_avg2 = 0
for number in range(n_test):
    noise1 = 0
    noise2 = 0
    for j in range(0,1400):
        noise1 = noise1 + outs1[number,j] * outs1[number,j]
        noise2 = noise2 + outs2[number,j] * outs2[number,j]
    noise1 = noise1 / 1400
    noise2 = noise2 / 1400
    max1 = 0
    max2 = 0
    for j in range(1600,2600):
        max1 = max1 + outs1[number,j] * outs1[number,j]
        max2 = max2 + outs2[number,j] * outs2[number,j]
    max1 = max1 / 1000
    max2 = max2 / 1000

    rr = abs(((max1/noise1)-1)/(max1/noise1))
    ideal_cc.append(math.sqrt(rr))

    ideal_cc_avg1 = ideal_cc_avg1 + math.sqrt(rr)
    rr = abs(((max2/noise2)-1)/(max2/noise2))
    ideal_cc.append(math.sqrt(rr))

    ideal_cc_avg2 = ideal_cc_avg2 + math.sqrt(rr)
    
    
    real_cc1,_ = stats.pearsonr(outs1[number,1600:2600],outs3[number,1600:2600])
    real_cc.append(real_cc1)
    real_cc2,_ = stats.pearsonr(outs2[number,1600:2600],outs4[number,1600:2600])
    real_cc.append(real_cc2)
    
print("AVG ideal_cc1:",ideal_cc_avg1/n_test)
print("AVG ideal_cc2:",ideal_cc_avg2/n_test)


# 从文件中读取数据 1
with h5py.File('data/cc_denoised_phase_diagram.h5', 'r') as f:
    real_cc_1 = f["real_cc"][:]
    ideal_cc_1 = f["ideal_cc"][:]


real_cc_2 = np.array(real_cc)
ideal_cc_2 = np.array(ideal_cc)


# 模型 1: real_cc_1 -> ideal_cc_1
b1 = real_cc_1.reshape(-1, 1)  # 转换为列向量
model1 = LinearRegression()
model1.fit(b1, ideal_cc_1)
slope1 = model1.coef_[0]
intercept1 = model1.intercept_
ideal_pred_1 = model1.predict(b1)
r2_1 = r2_score(ideal_cc_1, ideal_pred_1)

# 模型 2: real_cc_2 -> ideal_cc_2
b2 = real_cc_2.reshape(-1, 1)  # 转换为列向量
model2 = LinearRegression()
model2.fit(b2, ideal_cc_2)
slope2 = model2.coef_[0]
intercept2 = model2.intercept_
ideal_pred_2 = model2.predict(b2)
r2_2 = r2_score(ideal_cc_2, ideal_pred_2)

# 绘制合并图形
plt.figure(figsize=(8, 8))

# 数据集 1 (real_cc_1 -> ideal_cc_1)
plt.scatter(real_cc_1, ideal_cc_1, alpha=0.6, edgecolors='k', s=100, label="PD-Net")
#plt.plot(b1, ideal_pred_1, color='red', label=f'Fit 1: y = {slope1:.2f}x {intercept1:.2f}')

# 数据集 2 (ideal_cc_2 -> real_cc_2)
plt.scatter(real_cc_2, ideal_cc_2,color='red', alpha=0.6, edgecolors='k', s=100, label="Saad et al.")
#plt.plot(b2, ideal_pred_2, color='blue', label=f'Fit 2: y = {slope2:.2f}x')

# y=x 虚线
x = np.linspace(0.7, 1, 100)
plt.plot(x, x, linestyle='--', color='gray')

# 设置图形属性
plt.title("Denoising Effect", fontsize=24)
plt.xlabel("Real CC", fontsize=24)
plt.ylabel("Ideal CC", fontsize=24)
plt.xlim(0.7, 1)
plt.ylim(0.7, 1)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# 控制横纵坐标间隔和刻度字号
x_ticks = np.arange(0.7, 1.05, 0.1)  # 设置横坐标间隔
y_ticks = np.arange(0.7, 1.05, 0.1)  # 设置纵坐标间隔
plt.xticks(x_ticks, fontsize=22)      # 应用横坐标刻度和字号
plt.yticks(y_ticks, fontsize=22)      # 应用纵坐标刻度和字号
plt.legend(fontsize=22, frameon=True)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 8))
plt.scatter(corr_before, corr_after, alpha=0.7, edgecolors='k', s=50)
# 添加 y = x 的虚线
x = np.linspace(0.5, 1, 100)  # 定义 x 的范围
plt.plot(x, x, linestyle='--', color='gray')  # 添加虚线
plt.title("CC before and after denoising", fontsize=24)
plt.xlabel("Before", fontsize=20)
plt.ylabel("Afetr", fontsize=20)
plt.xlim(0.5, 1)  # 限制横坐标范围
plt.ylim(0.5, 1)  # 限制纵坐标范围
# 调整横纵坐标刻度的字号
plt.xticks(fontsize=20)  # 横坐标刻度字号
plt.yticks(fontsize=20)  # 纵坐标刻度字号
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()



# 读取数据
data_sets = [
    (corr_before, "CC(x,y)"),
    (corr_after, "CC(x',y')"),
]

# 绘制直方图
plt.figure(figsize=(8, 8))
bins = np.arange(0.2, 1.0 + 0.025, 0.05)  # 生成从 0.2 到 1.0，每隔 0.025 的区间

for data, label in data_sets:
    plt.hist(data, bins=bins, alpha=0.5, label=label, edgecolor='black')

# 图例和标签
plt.xlabel("CC", fontsize=24)
plt.ylabel('Frequency', fontsize=24)
plt.legend(fontsize=20, loc='upper left', frameon=False)  # 左上角，移除框线
plt.grid(axis='y', linestyle='--', alpha=0.7)
# 设置横纵坐标刻度字体大小
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0.2, 1)  # 限制横坐标范围
# 显示图形
plt.tight_layout()
plt.show()


with PdfPages('output_sadd_repeat.pdf') as pdf:
        
        for numberr in range(n_test):

            x=np.arange(200,2600)*0.025
            rf1=outs1[numberr,200:2600]
            rf2=outs3[numberr,200:2600]
            plt.figure(figsize=(8, 6))
            rf1 = np.reshape(rf1,(len(rf1)))
            rf2 = np.reshape(rf2,(len(rf2)))
            plt.plot(x,rf1,label="Repeat1 before")
            plt.plot(x,rf2,label="Repeat1 after")
            # 设置刻度显示间隔
            plt.ylim(-1.0, 1.0)
            plt.xticks(np.arange(5, 66, 10))  # x 轴从 5 到 65，每隔 5 显示一个刻度
            plt.yticks(np.arange(-1.0, 1.01, 0.5))  # y 轴从 -1 到 1.0，每隔 0.5 显示一个刻度
            # 调整刻度字体大小
            plt.tick_params(axis='both', which='major', labelsize=14)
            corr,_ = stats.pearsonr(rf1,rf2)
            plt.text(0.05, 0.05,'CC[%d]=%.3f'%(numberr,corr), transform=plt.gca().transAxes, verticalalignment='bottom',horizontalalignment='left')
            #print("Denoised.max",outs3[numberr,:].max())
            plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.,fontsize=14, frameon=False)
            pdf.savefig()
            plt.close()
            plt.show()
            
########## plot denoised repeat2  
            rf1=outs2[numberr,200:2600]
            rf2=outs4[numberr,200:2600]
            plt.figure(figsize=(8, 6))
            rf1 = np.reshape(rf1,(len(rf1)))
            rf2 = np.reshape(rf2,(len(rf2)))
            plt.plot(x,rf1,label="Repeat2 before")
            plt.plot(x,rf2,label="Repeat2 after")
            # 设置刻度显示间隔
            plt.ylim(-1.0, 1.0)
            plt.xticks(np.arange(5, 66, 10))  # x 轴从 5 到 65，每隔 5 显示一个刻度
            plt.yticks(np.arange(-1.0, 1.01, 0.5))  # y 轴从 -1 到 1.0，每隔 0.5 显示一个刻度
            # 调整刻度字体大小
            plt.tick_params(axis='both', which='major', labelsize=14)
            corr,_ = stats.pearsonr(rf1,rf2)
            plt.text(0.05, 0.05,'CC[%d]=%.3f'%(numberr,corr), transform=plt.gca().transAxes, verticalalignment='bottom',horizontalalignment='left')
            plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.,fontsize=14, frameon=False)
            pdf.savefig()
            plt.close()
            plt.show()

            rf1=outs1[numberr,200:2600]-outs3[numberr,200:2600]
            rf2=outs2[numberr,200:2600]-outs4[numberr,200:2600]
            plt.figure(figsize=(8, 6))
            rf1 = np.reshape(rf1,(len(rf1)))
            rf2 = np.reshape(rf2,(len(rf2)))
            plt.plot(x,rf1,label="Repeat1 noise")
            plt.plot(x,rf2,label="Repeat2 noise")
            # 设置刻度显示间隔
            plt.xticks(np.arange(5, 66, 10))  # x 轴从 5 到 65，每隔 5 显示一个刻度
            # 调整刻度字体大小
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.,fontsize=14, frameon=False)
            pdf.savefig()
            plt.close()
            plt.show()
            
########## Compare before denoised repeat 1&2  
            rf1=outs1[numberr,200:2600]
            rf2=outs2[numberr,i1[numberr]:i1[numberr]+2400]
            plt.figure(figsize=(8, 6))
            rf1 = np.reshape(rf1,(len(rf1)))
            rf2 = np.reshape(rf2,(len(rf2)))
            plt.plot(x,rf1,label="Repeat1 before")
            plt.plot(x,rf2,label="Repeat2 before")
            # 设置刻度显示间隔
            plt.ylim(-1.0, 1.0)
            plt.xticks(np.arange(5, 66, 10))  # x 轴从 5 到 65，每隔 5 显示一个刻度
            plt.yticks(np.arange(-1.0, 1.01, 0.5))  # y 轴从 -1 到 1.0，每隔 0.5 显示一个刻度
            # 调整刻度字体大小
            plt.tick_params(axis='both', which='major', labelsize=14)
            corr,_ = stats.pearsonr(rf1,rf2)
            plt.text(0.05, 0.05,'CC[%d]=%.3f'%(numberr,corr), transform=plt.gca().transAxes, verticalalignment='bottom',horizontalalignment='left')
            plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.,fontsize=14, frameon=False)
            pdf.savefig()
            plt.close()
            plt.show()
            
            x=np.arange(1600,2600)*0.025
            rf1=outs1[numberr,1600:2600]
            rf2=outs2[numberr,i1[numberr]+1400:i1[numberr]+2400]
            plt.figure(figsize=(8, 6))
            rf1 = np.reshape(rf1,(len(rf1)))
            rf2 = np.reshape(rf2,(len(rf2)))
            plt.plot(x,rf1,label="Repeat1 before")
            plt.plot(x,rf2,label="Repeat2 before")
            # 设置刻度显示间隔
            plt.ylim(-1.0, 1.0)
            plt.xticks(np.arange(40, 66, 5))  # x 轴从 5 到 65，每隔 5 显示一个刻度
            plt.yticks(np.arange(-1.0, 1.01, 0.5))  # y 轴从 -1 到 1.0，每隔 0.5 显示一个刻度
            # 调整刻度字体大小
            plt.tick_params(axis='both', which='major', labelsize=14)
            corr,_ = stats.pearsonr(rf1,rf2)
            plt.text(0.5, 0.05,'CC[%d]=%.3f'%(numberr,corr), transform=plt.gca().transAxes, verticalalignment='bottom',horizontalalignment='center')
            plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.,fontsize=14, frameon=False)
            pdf.savefig()
            plt.close()
            plt.show()
########## Compare after denoised repeat 1&2  

            x=np.arange(200,2600)*0.025
            rf1=outs3[numberr,200:2600]
            rf2=outs4[numberr,i2[numberr]:i2[numberr]+2400]
            plt.figure(figsize=(8, 6))
            rf1 = np.reshape(rf1,(len(rf1)))
            rf2 = np.reshape(rf2,(len(rf2)))
            plt.plot(x,rf1,label="Repeat1 after")
            plt.plot(x,rf2,label="Repeat2 after")
            # 设置刻度显示间隔
            plt.ylim(-1.0, 1.0)
            plt.xticks(np.arange(5, 66, 10))  # x 轴从 5 到 65，每隔 5 显示一个刻度
            plt.yticks(np.arange(-1.0, 1.01, 0.5))  # y 轴从 -1 到 1.0，每隔 0.5 显示一个刻度
            # 调整刻度字体大小
            plt.tick_params(axis='both', which='major', labelsize=14)
            corr,_ = stats.pearsonr(rf1,rf2)
            plt.text(0.05, 0.05,'CC[%d]=%.3f'%(numberr,corr), transform=plt.gca().transAxes, verticalalignment='bottom',horizontalalignment='left')
            plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.,fontsize=14, frameon=False)
            pdf.savefig()
            plt.close()
            
            x=np.arange(1600,2600)*0.025
            rf1=outs3[numberr,1600:2600]
            rf2=outs4[numberr,i2[numberr]+1400:i2[numberr]+2400]
            plt.figure(figsize=(8, 6))
            rf1 = np.reshape(rf1,(len(rf1)))
            rf2 = np.reshape(rf2,(len(rf2)))
            plt.plot(x,rf1,label="Repeat1 after")
            plt.plot(x,rf2,label="Repeat2 after")
            # 设置刻度显示间隔
            plt.ylim(-1.0, 1.0)
            plt.xticks(np.arange(40, 66, 5))  # x 轴从 5 到 65，每隔 5 显示一个刻度
            plt.yticks(np.arange(-1.0, 1.01, 0.5))  # y 轴从 -1 到 1.0，每隔 0.5 显示一个刻度
            # 调整刻度字体大小
            plt.tick_params(axis='both', which='major', labelsize=14)
            corr,_ = stats.pearsonr(rf1,rf2)
            plt.text(0.5, 0.05,'CC[%d]=%.3f'%(numberr,corr), transform=plt.gca().transAxes, verticalalignment='bottom',horizontalalignment='center')
            plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.,fontsize=14, frameon=False)
            pdf.savefig()
            plt.close()
    





