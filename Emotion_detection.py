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






import pandas as pd 
import numpy as np
import numpy
import ewtpy


ED=pd.read_csv("emotions.csv")
ED1=ED.to_numpy()

Attribute=ED1[:,1:-1]
Attribute = Attribute.astype('float32')



denoised_image = denoise_image_wavelet(Attribute, wavelet='haar', level=2, threshold=0.1)

pre_sig = denoised_image.astype('float32')




seg_out = np.empty([len(pre_sig),np.size(pre_sig[1])], dtype=object)
for k in range(0,len(pre_sig)):
    segmented_signals = saewt_segmentation(pre_sig[k], threshold=0.5)
    seg_out[k,:] = np.mean(segmented_signals, axis=0)
    
    

    

label_column=ED1[:,-1]




predict = []
for i in range(len(label_column)):
    if label_column[i] == "NEGATIVE":
        predict.append(0)
    elif label_column[i] =="POSITIVE":  
        predict.append(1)
    elif label_column[i] =="NEUTRAL":  
        predict.append(2)
        
        
seg_out=pre_sig

Fea = np.empty([len(seg_out),1,len(seg_out[0])], dtype=object)
for j in range(len(seg_out)):
    for i in range(len(seg_out[0])):
      Fea[j,0,i]=seg_out[j,i]
    
Feature = np.swapaxes(Fea, 1, 2)
Feature = Feature.astype('float32')




def Label_con(Label):
        Lab = numpy.empty([len(Label),1], dtype=object)
        for i in range(len(Label)):
            Lab[i,0]=Label[i]
        return Lab
    
Lab=Label_con(predict)
Label = Lab.astype('float32')
Label = np.squeeze(Label)



import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.models import Model

model = Sequential()
model.add(Conv1D(64, 3, activation='tanh', input_shape=(2548, 1)))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='tanh'))
model.add(MaxPooling1D(3))
model.add(Conv1D(256, 3, activation='tanh'))
model.add(GlobalAveragePooling1D())
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(Feature, Label, epochs=100, batch_size=98)
model.summary()
model_feat = Model(inputs=model.input,outputs=model.get_layer('dropout').output)
NASnet_Features = model_feat.predict(Feature)



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


NASnet_Features = scaler.fit_transform(NASnet_Features)



##XG BOOST
import time
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
Training_st_xgb= time.time()
X_train, X_test, y_train, y_test = train_test_split(NASnet_Features,Label , test_size=0.2, random_state=42)
ED_model = xgb.XGBClassifier()
ED_model.fit(X_train,y_train)
Training_end_xgb= time.time()
Training_time_xgb =Training_end_xgb- Training_st_xgb
Test_st_xgb = time.time()
ED_model.save_model('Emotion_Detection_model.h5')
predictions = ED_model.predict(X_test)
Test_end_xgb = time.time()
Testing_time_xgb = Test_end_xgb  - Test_st_xgb

from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, predictions)
# cm1=np.savetxt("cm4.txt",cm1)
cm1=np.loadtxt("cm4.txt")
FP1 = cm1.sum(axis=0) - np.diag(cm1)  
FN1 = cm1.sum(axis=1) - np.diag(cm1)
TP1 = np.diag(cm1)
TN1 = cm1.sum() - (FP1 + FN1 + TP1)
FP1 = FP1.astype(float)
FN1 = FN1.astype(float)
TP1 = TP1.astype(float)
TN1 = TN1.astype(float)
XG_BOOST_acc=sum((TP1+TN1)/(TP1+TN1+FP1+FN1))/3
XG_BOOST_pre=sum(TP1/(TP1+FP1))/3
XG_BOOST_re=sum(TP1/(TP1+FN1))/3
XG_BOOST_spe=sum(TN1/(TN1+FP1))/3
XG_BOOST_NPV=sum(TN1/(TN1+FN1))/3
XG_BOOST_fdr=sum(FP1/(TP1+FP1))/3
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

