import pandas as pd 
import numpy as np
import numpy


import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio

def soft_threshold(x, threshold):
    """ 
    Soft thresholding function.
    """
    return np.sign(x) * np.maximum(0, np.abs(x) - threshold)

def denoise_image_wavelet(image, wavelet='db1', level=1, threshold=0.1):
    """
    Denoise an image using wavelet soft thresholding.
    
    Args:
        image: Input image (2D array).
        wavelet: Wavelet to be used (default is 'db1').
        level: Decomposition level.
        threshold: Threshold value for soft thresholding.
    
    Returns:
        Denoised image.
    """
    # Perform 2D wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    # Apply soft thresholding to the detail coefficients
    coeffs_thresh = [coeffs[0]]
    for detail_level in range(1, len(coeffs)):
        coeff_arr = coeffs[detail_level]
        coeffs_thresh.append(tuple(soft_threshold(c, threshold) for c in coeff_arr))

    # Reconstruct the image from the denoised coefficients
    denoised_image = pywt.waverec2(coeffs_thresh, wavelet)
    
    return denoised_image

import numpy as np
import matplotlib.pyplot as plt
import ewtpy

def saewt_segmentation(signal, threshold=0.1):
    """
    Perform Spectrum-Adaptive Segmentation Empirical Wavelet Transform (SAEWT)
    for signal segmentation.

    Args:
        signal: Input signal (1D array).
        threshold: Threshold value for spectrum-based adaptive segmentation.

    Returns:
        Segmented signals.
    """
    # Perform EWT
    ewt, _, _ = ewtpy.EWT1D(signal, N=2)

    # Calculate power spectrum
    power_spectrum = np.abs(ewt)**2

    # Apply spectrum-based adaptive segmentation
    segmented_signals = []
    for i in range(power_spectrum.shape[1]):
        mask = power_spectrum[:, i] > threshold
        segmented_signal = ewt[:, i] * mask
        segmented_signals.append(segmented_signal)

    return np.array(segmented_signals)



ESD=pd.read_csv("Epileptic_seizures_dataset.csv")
ESD1=ESD.to_numpy()
ESD_Attribute=ESD1[:,1:-1]
ESD_Attribute = ESD_Attribute.astype('float32')


noisy_image = ESD_Attribute + 0.5 * np.random.randn(*ESD_Attribute.shape)  # Add some Gaussian noise

# Denoise the image
denoised_image = denoise_image_wavelet(noisy_image, wavelet='haar', level=2, threshold=0.1)

Denoised = denoised_image.astype('float32')


seg_out = np.empty([len(Denoised),np.size(Denoised[1])], dtype=object)


for k in range(0,len(Denoised)):
    segmented_signals = saewt_segmentation(Denoised[k], threshold=0.5)
    seg_out[k,:] = np.mean(segmented_signals, axis=0)



ESD_Attribute=seg_out



ESD_label_column=ESD1[:,-1]


ESD_predict = []
for i in range(len(ESD_label_column)):
    if ESD_label_column[i] == 1:
        ESD_predict.append(0)
    elif ESD_label_column[i] in [2, 3, 4, 5]:  
        ESD_predict.append(1)

ESD_Fea = np.empty([len(ESD_Attribute),1,len(ESD_Attribute[0])], dtype=object)
for j in range(len(ESD_Attribute)):
    for i in range(len(ESD_Attribute[0])):
      ESD_Fea[j,0,i]=ESD_Attribute[j,i]
    
ESD_Feature = np.swapaxes(ESD_Fea, 1, 2)
ESD_Feature = ESD_Feature.astype('float32')




def Label_con(Label):
        Lab = numpy.empty([len(Label),1], dtype=object)
        for i in range(len(Label)):
            Lab[i,0]=Label[i]
        return Lab
    
ESD_Lab=Label_con(ESD_predict)
ESD_Label = ESD_Lab.astype('float32')
ESD_Label = np.squeeze(ESD_Label)





import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.models import Model

model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(178, 1)))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(256, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(ESD_Feature, ESD_Label, epochs=5, batch_size=64)
model.summary()
model_feat = Model(inputs=model.input,outputs=model.get_layer('dense').output)
NASnet_Features = model_feat.predict(ESD_Feature)


##XG BOOST
import time
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
Training_st_xgb= time.time()
X_train, X_test, y_train, y_test = train_test_split(NASnet_Features,ESD_Label , test_size=0.2, random_state=42)
model = xgb.XGBClassifier()
model.fit(X_train,y_train)
Training_end_xgb= time.time()
Training_time_MLP =Training_end_xgb- Training_st_xgb
Test_st_xgb = time.time()
predictions = model.predict(X_test)
Test_end_xgb = time.time()
Testing_time_xgb = Test_end_xgb  - Test_st_xgb

from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, predictions)
# cm1=np.loadtxt("cm1.txt")
FP1 = cm1.sum(axis=0) - np.diag(cm1)  
FN1 = cm1.sum(axis=1) - np.diag(cm1)
TP1 = np.diag(cm1)
TN1 = cm1.sum() - (FP1 + FN1 + TP1)
FP1 = FP1.astype(float)
FN1 = FN1.astype(float)
TP1 = TP1.astype(float)
TN1 = TN1.astype(float)
XG_BOOST_acc=sum((TP1+TN1)/(TP1+TN1+FP1+FN1))/2
XG_BOOST_pre=sum(TP1/(TP1+FP1))/2
XG_BOOST_re=sum(TP1/(TP1+FN1))/2
XG_BOOST_spe=sum(TN1/(TN1+FP1))/2
XG_BOOST_NPV=sum(TN1/(TN1+FN1))/2
XG_BOOST_fdr=sum(FP1/(TP1+FP1))/2
XG_BOOST_FNR=1-XG_BOOST_re
XG_BOOST_FPR=1-XG_BOOST_spe 
XG_BOOST_fs = 2 * (XG_BOOST_pre * XG_BOOST_re) / (XG_BOOST_pre +XG_BOOST_re)  
XG_BOOST_PLR=XG_BOOST_re/(1-XG_BOOST_spe)
XG_BOOST_mcc=matthews_corrcoef(y_test, predictions)
XG_BOOST_kappa=cohen_kappa_score(y_test, predictions)



from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
csfont = {'fontname':'Times New Roman'}
fig, ax = plot_confusion_matrix(conf_mat=cm1)
plt.ylabel('True Label', fontsize=16,**csfont)
plt.xlabel('Predicted Label', fontsize=16,**csfont)
plt.show()