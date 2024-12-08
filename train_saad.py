# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:23:34 2024

@author: Administrator
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from scipy import signal
import matplotlib.pyplot as plt
from Utils import *
import numpy as np
import h5py
from scipy import stats

output_nz = 2801
n_test = 500


f = h5py.File('data/individual_test.h5', 'r')
test_input = f["data"][:,:]
f.close()

f = h5py.File('data/compare_test.h5','w')   #创建一个h5文件，文件指针是f
f['data1'] = test_input[0:500,1:]              #将数据写入文件的主键data下面
f['data2'] = test_input[500:n_test,1:]          #将数据写入文件的主键labels下面
f.close()


outs1 = np.zeros((n_test*2, 2800))

for nnn in range(n_test*2):
    dn = test_input[nnn,:]
    dn =  butter_bandpass_filter_zi(dn, 1, 45, 100, order=10)
    fs = 40
    le = 20
    leov = int(le/2)
    f, t, Zxx = signal.stft(dn, fs=fs, window='hann', nperseg=le, noverlap=leov, nfft=2*le, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=- 1)
    dnPatchR = np.real(Zxx)
    dnPatchI = np.imag(Zxx)
    dnPatchA = np.abs(Zxx)

    if np.max(dnPatchA)>0:
        maA = np.max(np.abs(dnPatchA)) 
        maR = np.max(np.abs(dnPatchR)) 
        maI = np.max(np.abs(dnPatchI)) 
        dnPatchR = dnPatchR/np.max(np.abs(dnPatchR))
        dnPatchI = dnPatchI/np.max(np.abs(dnPatchI))
        dnPatchA = dnPatchA/np.max(np.abs(dnPatchA))
    w1 =8
    w2 =8
    s1z =1
    s2z =1
    dn_patchA = yc_patch(dnPatchA,w1,w2,s1z,s2z)
    dn_patchR = yc_patch(dnPatchR,w1,w2,s1z,s2z)
    dn_patchI = yc_patch(dnPatchI,w1,w2,s1z,s2z)
    from keras.layers import Input, Dense, Dropout, Add, multiply, GlobalAvgPool1D, Reshape, concatenate, UpSampling1D, Flatten, MaxPooling1D, BatchNormalization, average, Conv1D
    from keras.models import Model
    import scipy.io
    from keras import optimizers
    from keras.callbacks import EarlyStopping
    import h5py
    from math import*
    import numpy as np
    from keras.callbacks import EarlyStopping,ModelCheckpoint
    from keras import backend as K
    import tensorflow as tf

    import oct2py
    import numpy as np
    import h5py
    #oc = oct2py.Oct2Py()

    corr1=0
    corr2=1
    corr3=1
    #dataNoise = np.load(r'E:\1DDenoising/dn_patch_SCALO.npy')
    dataNoiseR = dn_patchR
    dataNoiseI = dn_patchI
    dataNoiseA = dn_patchA

    print(np.shape(dataNoiseR))

    # Random Permute for the patches for training
    #ind = np.random.permutation(len(dataNoise))
    #dataNoise = np.array(dataNoise)
    #dataNoise = dataNoise[ind]

    # Orders.
    D1 = 32
    D2 = int(D1 / 2)
    D3 = int(D2 / 2)
    D4 = int(D3 / 2)
    D5 = int(D4 / 2)
    D6 = int(D5 / 2)

    # Real and Imaginary Inputs.
    INPUT_SIZE1 = dataNoiseR.shape[0]  # 数据的样本数量
    INPUT_SIZE2 = dataNoiseR.shape[1]  # 数据的特征数

    # Input Layers with defined shape
    input_img1 = Input(shape=(INPUT_SIZE2,))
    input_img2 = Input(shape=(INPUT_SIZE2,))

    # Reshape inputs to ensure compatibility
    input_img1x = Reshape((INPUT_SIZE2, 1))(input_img1)
    input_img2x = Reshape((INPUT_SIZE2, 1))(input_img2)

    # Concatenate the real and imaginary parts
    input_img3 = concatenate([input_img1x, input_img2x])

    # Define compactlayer
    def compactlayer(y, D):
        s0 = Lambda(lambda x: x[:, :, 0])(y)
        s1 = Lambda(lambda x: x[:, :, 1])(y)

        B1 = Block(s0, D)
        B2 = Block(s1, D)
        
        # Concatenate and add attention layer (assuming Block and Attention layers are defined elsewhere)
        B = concatenate([B1, B2], axis=-1)
        Batt = Attention(B)
        
        return Batt

    # Encoder部分
    e1 = compactlayer(input_img3, D1)
    e2 = compactlayer(e1, D2)
    e3 = compactlayer(e2, D3)

    # Decoder部分
    d4 = compactlayer(e3, D2)
    d4 = Add()([d4, e2])
    d5 = compactlayer(d4, D1)
    d5 = Add()([d5, e1])
    d6 = compactlayer(d5, D1)

    # Flatten and output
    d6 = Flatten()(d6)
    outA = Dense(INPUT_SIZE2, activation='softplus')(d6)




    # Customize Loss Function.
    def MASK_LOSS(y_true, y_pred):
        
        nois = y_true - y_pred
        maskbiS =  1 / (1 + (K.abs(nois)/K.abs(y_pred)))
        maskbiN = (K.abs(nois)/K.abs(y_pred)) / (1+(K.abs(nois)/K.abs(y_pred)))
        
        #return   K.abs(1-maskbiS) + K.abs(maskbiN)   
        return   K.abs(1-maskbiS) 

                
    autoencoder = Model([input_img1,input_img2], [outA])
    sgd = optimizers.Adam(lr=0.0001)
    autoencoder.compile(optimizer=sgd, loss=[MASK_LOSS])    



    autoencoder.summary()

    # Addig stopping condition
    es=EarlyStopping(monitor='loss',mode='min',verbose=1,patience=5)
    mc=ModelCheckpoint('best_model_STFT_STEAD_Real_10.h5',monitor='loss',mode='min',save_best_only=True)
    batch = int(np.ceil(len(dataNoiseR)*0.005))
    #batch = 128
    print('Batch Size is:' + str(batch))
    tic()
    history = autoencoder.fit([dataNoiseR,dataNoiseI],[dataNoiseA], epochs=50, batch_size=batch, shuffle=False,callbacks=[es,mc],verbose=0)
    toc()
    # Predict
    from keras.models import load_model

    batch = batch

    model = load_model('best_model_STFT_STEAD_Real_10.h5', custom_objects={"MASK_LOSS":MASK_LOSS})
    print('Loaded Best Model')

    tic()
    xA = model.predict([dataNoiseR,dataNoiseI],batch_size=batch)
    toc()
    # Unpatching
    X1 = np.transpose(xA)
    n1,n2=np.shape(dnPatchA)
    outA = yc_patch_inv(X1,n1,n2,w1,w2,s1z,s2z)

    outA = np.array(outA)

    #Plot Input Target and Predicted Mask.
    plt.imshow(dnPatchA, aspect='auto')
    plt.figure()
    plt.imshow(outA, aspect='auto')
    plt.show()
    dn = test_input[nnn,:]
    f, t, Zxxdn = signal.stft(dn, fs=fs, window='hann', nperseg=le, noverlap=leov, nfft=2*le, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=- 1)
    ZxxdnR = np.real(Zxxdn)
    ZxxdnI = np.imag(Zxxdn)



    yy = 1
    sizmedian = 15
    outAM = (outA*maA)



    outAM = outAM/np.max(np.abs(outAM))

    for iu in range(outAM.shape[0]):
        tmp = np.copy(outAM[iu,:])
        #tmp[tmp>=1*np.mean(tmp)]=1
        tmp[tmp<1*np.mean(tmp)]=1e-10
        outAM[iu,:] = tmp


    mea = np.mean(outAM)
    outAM[outAM>=yy*mea]=1
    outAM[outAM<yy*mea]=1e-10
    outAM = ndimage.median_filter(outAM, size=sizmedian)


    dn_mask = np.real(Zxxdn)*outAM + 1j*np.imag(Zxxdn)*outAM


    rec = signal.istft(dn_mask,  fs=fs, window='hann', nperseg=le, noverlap=leov, nfft=2*le, boundary='zeros')
    rec = np.array(rec)
    outs1[nnn,:] = rec[1,:]
    print("%d finished",nnn)


f = h5py.File('saad_result.h5','w')   #创建一个h5文件，文件指针是f
f['data1'] = outs1[:n_test,:]              #将数据写入文件的主键data下面
f['data2'] = outs1[n_test:,:]          #将数据写入文件的主键labels下面
f.close()




