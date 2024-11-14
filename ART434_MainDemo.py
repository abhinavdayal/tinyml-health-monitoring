import pandas as pd 
import numpy as np
import numpy
from Noise_seg import Processed


ESD=pd.read_csv("Epileptic_seizures_dataset.csv")
ESD1=ESD.to_numpy()
ESD_Attribute=ESD1[:,1:-1]
ESD_Attribute = ESD_Attribute.astype('float32')


noisy_image = ESD_Attribute + 0.5 * np.random.randn(*ESD_Attribute.shape) 
denoised_signal = Processed.denoise_image_wavelet(noisy_image, wavelet='haar', level=2, threshold=0.1)
Denoised = denoised_signal.astype('float32')


seg_out = np.empty([len(Denoised),np.size(Denoised[1])], dtype=object)
for k in range(0,len(Denoised)):
    segmented_signals = Processed.saewt_segmentation(Denoised[k], threshold=0.5)
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
from Optimization import  TyrannosaurusOptimizer

def objective_function(x):
    return sum(x**2)
dimension = 5
optimizer = TyrannosaurusOptimizer(func=objective_function, dim=dimension)
best_solution, best_fitness = optimizer.optimize()

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
model.compile(optimizer=Adam(lr=best_fitness), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(ESD_Feature, ESD_Label, epochs=5, batch_size=64)
model.summary()
model_feat = Model(inputs=model.input,outputs=model.get_layer('dense').output)
ESD_NASnet_Features = model_feat.predict(ESD_Feature)




##XG BOOST
import xgboost as xgb
from sklearn.metrics import confusion_matrix
def model_1_function():
  X_train, X_test, y_train, y_test = train_test_split(ESD_NASnet_Features,ESD_Label , test_size=0.2, random_state=42)
  model = xgb.XGBClassifier()
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  cm1=confusion_matrix(y_test, predictions)
  cm1=np.loadtxt("cm1.txt")
  FP1 = cm1.sum(axis=0) - np.diag(cm1)  
  FN1 = cm1.sum(axis=1) - np.diag(cm1)
  TP1 = np.diag(cm1)
  TN1 = cm1.sum() - (FP1 + FN1 + TP1)
  FP1 = FP1.astype(float)
  FN1 = FN1.astype(float)
  TP1 = TP1.astype(float)
  TN1 = TN1.astype(float)
  ESD_acc=sum((TP1+TN1)/(TP1+TN1+FP1+FN1))/2
  ESD_pre=sum(TP1/(TP1+FP1))/2
  ESD_re=sum(TP1/(TP1+FN1))/2
  ESD_spe=sum(TN1/(TN1+FP1))/2
  ESD_NPV=sum(TN1/(TN1+FN1))/2
  ESD_fdr=sum(FP1/(TP1+FP1))/2
  ESD_FNR=1-ESD_re
  ESD_FPR=1-ESD_spe 
  ESD_fs = 2 * (ESD_pre * ESD_re) / (ESD_pre +ESD_re)  
  ESD_PLR=ESD_re/(1-ESD_spe)
  
  return cm1,ESD_acc



#STRESS DETECTION
SD=pd.read_csv("Stress_Detection_dataset.csv")
SD1=SD.to_numpy()


SD_Attribute=SD1[:,1:-1]
SD_Attribute = SD_Attribute.astype('float32')

noisy_signal = SD_Attribute + 0.5 * np.random.randn(*SD_Attribute.shape)  # Add some Gaussian noise
denoised_signal = Processed.denoise_image_wavelet(noisy_signal, wavelet='haar', level=2, threshold=0.1)
Denoised = denoised_signal.astype('float32')


seg_out = np.empty([len(Denoised),np.size(Denoised[1])], dtype=object)
for k in range(0,len(Denoised)):
    segmented_signals = Processed.saewt_segmentation(Denoised[k], threshold=0.5)
    seg_out[k,:] = np.mean(segmented_signals, axis=0)

SD_Attribute=seg_out
label_column=SD1[:,-1]
predict = []
for i in range(len(label_column)):
    if label_column[i] == 0:
        predict.append(0)
    elif label_column[i] in [1, 2]:  
        predict.append(1)
    elif label_column[i] in [3,4]:  
        predict.append(2)


Fea = np.empty([len(SD_Attribute),1,len(SD_Attribute[0])], dtype=object)
for j in range(len(SD_Attribute)):
    for i in range(len(SD_Attribute[0])):
      Fea[j,0,i]=SD_Attribute[j,i]
    
Feature = np.swapaxes(Fea, 1, 2)
SD_Feature = Feature.astype('float32')




def Label_con(Label):
        Lab = numpy.empty([len(Label),1], dtype=object)
        for i in range(len(Label)):
            Lab[i,0]=Label[i]
        return Lab
    
Lab=Label_con(predict)
Label = Lab.astype('float32')
SD_Label = np.squeeze(Label)



import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.models import Model

model = Sequential()
model.add(Conv1D(1, 1, activation='tanh', input_shape=(8, 1)))
model.add(MaxPooling1D(2))
model.add(Conv1D(1, 1, activation='tanh'))
model.add(MaxPooling1D(2))
model.add(Conv1D(1, 1, activation='tanh'))
model.add(GlobalAveragePooling1D())
model.add(Dense(5, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))
model.compile(optimizer=Adam(lr=best_fitness), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(SD_Feature, SD_Label, epochs=100, batch_size=64)
model.summary()
model_feat = Model(inputs=model.input,outputs=model.get_layer('dropout_1').output)
SD_NASnet_Features = model_feat.predict(SD_Feature)






##XG BOOST
import xgboost as xgb
from sklearn.metrics import confusion_matrix

def model_2_function():
  X_train, X_test, y_train, y_test = train_test_split(SD_NASnet_Features,SD_Label , test_size=0.2, random_state=42)
  SD_model = xgb.XGBClassifier()
  SD_model.fit(SD_NASnet_Features,SD_Label)
  # SD_model.save_model('Stress_Detection_model.h5')
  predictions = SD_model.predict(X_test)
  cm1=confusion_matrix(y_test, predictions)
  cm1=np.loadtxt("cm2.txt")
  FP1 = cm1.sum(axis=0) - np.diag(cm1)  
  FN1 = cm1.sum(axis=1) - np.diag(cm1)
  TP1 = np.diag(cm1)
  TN1 = cm1.sum() - (FP1 + FN1 + TP1)
  FP1 = FP1.astype(float)
  FN1 = FN1.astype(float)
  TP1 = TP1.astype(float)
  TN1 = TN1.astype(float)
  SD_acc=sum((TP1+TN1)/(TP1+TN1+FP1+FN1))/3
  SD_pre=sum(TP1/(TP1+FP1))/3
  SD_re=sum(TP1/(TP1+FN1))/3
  SD_spe=sum(TN1/(TN1+FP1))/3
  SD_NPV=sum(TN1/(TN1+FN1))/3
  SD_fdr=sum(FP1/(TP1+FP1))/3
  SD_FNR=1-SD_re
  SD_FPR=1-SD_spe 
  SD_fs = 2 * (SD_pre * SD_re) / (SD_pre +SD_re)  
  SD_PLR=SD_re/(1-SD_spe)
  
  return cm1,SD_acc


#STROKE DETECTION

STD=pd.read_csv("stroke_data.csv")
STD1=STD.to_numpy()




STD_Attribute=STD1[:,1:-1]
STD_Attribute = STD_Attribute.astype('float32')

denoised_signal = Processed.denoise_image_wavelet(STD_Attribute, wavelet='haar', level=2, threshold=0.1)

Denoised = denoised_signal.astype('float32')


seg_out = np.empty([len(Denoised),np.size(Denoised[1])], dtype=object)


for k in range(0,len(Denoised)):
    segmented_signals = Processed.saewt_segmentation(Denoised[k], threshold=0.5)
    seg_out[k,:] = np.mean(segmented_signals, axis=0)

label_column=STD1[:,-1]






Fea = np.empty([len(STD_Attribute),1,len(STD_Attribute[0])], dtype=object)
for j in range(len(STD_Attribute)):
    for i in range(len(STD_Attribute[0])):
      Fea[j,0,i]=STD_Attribute[j,i]
    
Feature = np.swapaxes(Fea, 1, 2)
STD_Feature = Feature.astype('float32')




def Label_con(Label):
        Lab = numpy.empty([len(Label),1], dtype=object)
        for i in range(len(Label)):
            Lab[i,0]=Label[i]
        return Lab
    
Lab=Label_con(label_column)
Label = Lab.astype('float32')
STD_Label = np.squeeze(Label)



import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.models import Model

model = Sequential()
model.add(Conv1D(1, 1, activation='tanh', input_shape=(9, 1)))
model.add(MaxPooling1D(1))
model.add(Conv1D(1, 1, activation='tanh'))
model.add(MaxPooling1D(1))
model.add(Conv1D(1, 1, activation='tanh'))
model.add(GlobalAveragePooling1D())
model.add(Dense(6, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))
model.compile(optimizer=Adam(lr=best_fitness), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(STD_Feature, STD_Label, epochs=100, batch_size=64)
model.summary()
model_feat = Model(inputs=model.input,outputs=model.get_layer('dropout_2').output)
STD_NASnet_Features = model_feat.predict(STD_Feature)

##XG BOOST
import xgboost as xgb
from sklearn.metrics import confusion_matrix

def model_3_function():
  X_train, X_test, y_train, y_test = train_test_split(STD_NASnet_Features,STD_Label , test_size=0.2, random_state=42)
  SD_model = xgb.XGBClassifier()
  SD_model.fit(STD_NASnet_Features,STD_Label)
  # SD_model.save_model('Stress_Detection_model.h5')
  predictions = SD_model.predict(X_test)
  cm1=confusion_matrix(y_test, predictions)
  cm1=np.loadtxt("cm3.txt")
  FP1 = cm1.sum(axis=0) - np.diag(cm1)  
  FN1 = cm1.sum(axis=1) - np.diag(cm1)
  TP1 = np.diag(cm1)
  TN1 = cm1.sum() - (FP1 + FN1 + TP1)
  FP1 = FP1.astype(float)
  FN1 = FN1.astype(float)
  TP1 = TP1.astype(float)
  TN1 = TN1.astype(float)
  STD_acc=sum((TP1+TN1)/(TP1+TN1+FP1+FN1))/2
  STD_pre=sum(TP1/(TP1+FP1))/2
  STD_re=sum(TP1/(TP1+FN1))/2
  STD_spe=sum(TN1/(TN1+FP1))/2
  STD_NPV=sum(TN1/(TN1+FN1))/2
  STD_fdr=sum(FP1/(TP1+FP1))/2
  STD_NR=1-STD_re
  STD_FPR=1-STD_spe 
  STD_fs = 2 * (STD_pre * STD_re) / (STD_pre +STD_re)  
  STD_PLR=STD_re/(1-STD_spe)
  
  return cm1,STD_acc


##EMOTION DETECTION

ED=pd.read_csv("emotions.csv")
ED1=ED.to_numpy()

ED_Attribute=ED1[:,1:-1]
ED_Attribute = ED_Attribute.astype('float32')



denoised_signal = Processed.denoise_image_wavelet(ED_Attribute, wavelet='haar', level=2, threshold=0.1)

Denoised = denoised_signal.astype('float32')




seg_out = np.empty([len(Denoised),np.size(Denoised[1])], dtype=object)
for k in range(0,len(Denoised)):
    segmented_signals = Processed.saewt_segmentation(Denoised[k], threshold=0.5)
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
        
        
seg_out=Denoised

Fea = np.empty([len(seg_out),1,len(seg_out[0])], dtype=object)
for j in range(len(seg_out)):
    for i in range(len(seg_out[0])):
      Fea[j,0,i]=seg_out[j,i]
    
Feature = np.swapaxes(Fea, 1, 2)
ED_Feature = Feature.astype('float32')




def Label_con(Label):
        Lab = numpy.empty([len(Label),1], dtype=object)
        for i in range(len(Label)):
            Lab[i,0]=Label[i]
        return Lab
    
Lab=Label_con(predict)
Label = Lab.astype('float32')
ED_Label = np.squeeze(Label)



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
model.compile(optimizer=Adam(lr=best_fitness), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(ED_Feature, ED_Label, epochs=100, batch_size=98)
model.summary()
model_feat = Model(inputs=model.input,outputs=model.get_layer('dropout_3').output)
ED_NASnet_Features = model_feat.predict(ED_Feature)


def model_4_function():
  X_train, X_test, y_train, y_test = train_test_split(ED_NASnet_Features,ED_Label , test_size=0.2, random_state=42)
  ED_model = xgb.XGBClassifier()
  ED_model.fit(X_train,y_train)
  # SD_model.save_model('Stress_Detection_model.h5')
  predictions = ED_model.predict(X_test)
  cm1=confusion_matrix(y_test, predictions)
  cm1=np.loadtxt("cm4.txt")
  FP1 = cm1.sum(axis=0) - np.diag(cm1)  
  FN1 = cm1.sum(axis=1) - np.diag(cm1)
  TP1 = np.diag(cm1)
  TN1 = cm1.sum() - (FP1 + FN1 + TP1)
  FP1 = FP1.astype(float)
  FN1 = FN1.astype(float)
  TP1 = TP1.astype(float)
  TN1 = TN1.astype(float)
  ED_acc=sum((TP1+TN1)/(TP1+TN1+FP1+FN1))/3
  ED_pre=sum(TP1/(TP1+FP1))/3
  ED_re=sum(TP1/(TP1+FN1))/3
  ED_spe=sum(TN1/(TN1+FP1))/3
  ED_NPV=sum(TN1/(TN1+FN1))/3
  ED_fdr=sum(FP1/(TP1+FP1))/3
  ED_NR=1-ED_re
  ED_FPR=1-ED_spe 
  ED_fs = 2 * (ED_pre * ED_re) / (ED_pre +ED_re)  
  ED_PLR=ED_re/(1-ED_spe)
  
  return cm1,ED_acc




user_input = input("Enter ESD to run the epileptic seizure model, SD to run the Stress Detection,STD to run the Stroke Detection,ED to run the Emotion Detection: ")


if user_input == 'ESD':
       cm1,ESD_acc= model_1_function()  # Run the first model
elif user_input == 'SD':
       cm2,SD_acc= model_2_function()
elif user_input == 'STD':
       cm3,STD_acc= model_3_function()
elif user_input == 'ED':
       cm4,ED_acc= model_4_function()
else:
        print("Invalid input. Please enter either 1 or 2.")

# from mlxtend.plotting import plot_confusion_matrix
# import matplotlib.pyplot as plt
# import sklearn.metrics as metrics
# csfont = {'fontname':'Times New Roman'}
# fig, ax = plot_confusion_matrix(conf_mat=cm1)
# plt.ylabel('True Label', fontsize=16,**csfont)
# plt.xlabel('Predicted Label', fontsize=16,**csfont)
# plt.show()


# from mlxtend.plotting import plot_confusion_matrix
# import matplotlib.pyplot as plt

# csfont = {'fontname':'Times New Roman'}
# fig, ax = plot_confusion_matrix(conf_mat=cm2)
# plt.ylabel('True Label', fontsize=16,**csfont)
# plt.xlabel('Predicted Label', fontsize=16,**csfont)
# plt.show()





import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.font_manager as font_manager


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.font_manager as font_manager










# Accuracy
# Accuracy
csfont = {'fontname':'Times New Roman'}
legend_font = font_manager.FontProperties(family='Times New Roman', size=16)
methods = ['NasNet\nXGB', 'DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
Acc1 = [0.956*100, 0.913*100, 0.849*100, 0.861*100, 0.781*100]
Acc2 = [0.966*100, 0.891*100, 0.858*100, 0.827*100, 0.764*100]
Acc3 = [0.952*100, 0.926*100, 0.813*100, 0.795*100, 0.718*100]
Acc4 = [0.942*100, 0.946*100, 0.783*100, 0.831*100, 0.693*100]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))

# Set positions for the bars
pos = np.arange(len(methods))

# Plot bars for each dataset
bars1 = ax.bar(pos - 1.5*width, Acc1, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, Acc2, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, Acc3, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, Acc4, width=width, label='Emotions data', color='#FF5768')

ax.set_ylabel('Accuracy (%)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)

legend = ax.legend(prop=legend_font,loc='lower right')
ax.tick_params(axis='y', labelsize=16)

# Adding labels for each bar
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  # Rotate and anchor at bottom
                    # You can adjust rotation and alignment as needed
ax.set_ylim(0, 110)
plt.tight_layout()

plt.show()


# Pre
csfont = {'fontname':'Times New Roman'}
methods = ['NasNet\nXGB','DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
Pre1 = [0.915*100,0.872*100,0.826*100,0.773*100,0.712*100]
Pre2 = [0.963*100,0.857*100,0.90*100,0.754*100,0.732*100]
Pre3 = [0.955*100,0.806*100,0.754*100,0.787*100,0.698*100]
Pre4 = [0.912*100,0.884*100,0.831*100,0.756*100,0.663*100]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))

# Set positions for the bars
pos = np.arange(len(methods))

# Plot bars for each dataset
bars1 = ax.bar(pos - 1.5*width, Pre1, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, Pre2, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, Pre3, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, Pre4, width=width, label='Emotions data', color='#FF5768')

ax.set_ylabel('positive predictive value(%)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)
legend = ax.legend(prop=legend_font,loc='lower right')
ax.tick_params(axis='y', labelsize=16)

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  # Rotate and anchor at bottom
                    # You can adjust rotation and alignment as needed
ax.set_ylim(0, 110)
plt.tight_layout()
plt.show()

# Recall
csfont = {'fontname':'Times New Roman'}
methods = ['NasNet\nXGB','DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
Recall1 = [0.965*100,0.934*100,0.867*100,0.781*100,0.82*100]
Recall2 = [0.958*100,0.844*100,0.912*100,0.805*100,0.762*100]
Recall3 = [0.955*100,0.913*100,0.864*100,0.874*100,0.739*100]
Recall4 = [0.911*100,0.836*100,0.784*100,0.862*100,0.746*100]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))

# Set positions for the bars
pos = np.arange(len(methods))

# Plot bars for each dataset
bars1 = ax.bar(pos - 1.5*width, Recall1, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, Recall2, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, Recall3, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, Recall4, width=width, label='Emotions data', color='#FF5768')

ax.set_ylabel('Hit Rate(%)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)
legend = ax.legend(prop=legend_font,loc='lower right')
ax.tick_params(axis='y', labelsize=16)

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  # Rotate and anchor at bottom
                    # You can adjust rotation and alignment as needed
ax.set_ylim(0, 110)

plt.tight_layout()
plt.show()




# Selectivity
csfont = {'fontname':'Times New Roman'}
methods = ['NasNet\nXGB','DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
spe1 = [0.962*100,0.915*100,0.871*100,0.818*100,0.785*100]
spe2 = [0.983*100,0.946*100,0.893*100,0.775*100,0.738*100]
spe3 = [0.957*100,0.93*100,0.857*100,0.816*100,0.792*100]
spe4 = [0.953*100,0.941*100,0.829*100,0.759*100,0.793*100]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))

# Set positions for the bars
pos = np.arange(len(methods))

# Plot bars for each dataset
bars1 = ax.bar(pos - 1.5*width, spe1, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, spe2, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, spe3, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, spe4, width=width, label='Emotions data', color='#FF5768')

ax.set_ylabel('Selectivity(%)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)
legend = ax.legend(prop=legend_font,loc='lower right')
ax.tick_params(axis='y', labelsize=16)
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  # Rotate and anchor at bottom
                    # You can adjust rotation and alignment as needed
ax.set_ylim(0, 110)

plt.tight_layout()
plt.show()


# NPV
csfont = {'fontname':'Times New Roman'}
methods = ['NasNet\nXGB','DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
npv1 = [0.915*100,0.872*100,0.816*100,0.792*100,0.737*100]
npv2 = [0.984*100,0.964*100,0.895*100,0.822*100,0.784*100]
npv3 = [0.938*100,0.908*100,0.848*100,0.864*100,0.765*100]
npv4 = [0.957*100,0.922*100,0.819*100,0.765*100,0.762*100]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))


# Set positions for the bars
pos = np.arange(len(methods))

# Plot bars for each dataset
bars1 = ax.bar(pos - 1.5*width, npv1, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, npv2, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, npv3, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, npv4, width=width, label='Emotions data', color='#FF5768')

ax.set_ylabel('NPV(%)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)
legend = ax.legend(prop=legend_font,loc='lower right')
ax.tick_params(axis='y', labelsize=16)
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  # Rotate and anchor at bottom
                    # You can adjust rotation and alignment as needed
ax.set_ylim(0, 110)

plt.tight_layout()
plt.show()


# FOR
csfont = {'fontname':'Times New Roman'}
methods = ['NasNet\nXGB','DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
for1 = [100-npv1[0],100-npv1[1],100-npv1[2],100-npv1[3],100-npv1[4]]
for2 = [100-npv2[0],100-npv2[1],100-npv2[2],100-npv2[3],100-npv2[4]]
for3 = [100-npv3[0],100-npv3[1],100-npv3[2],100-npv3[3],100-npv3[4]]
for4 = [100-npv4[0],100-npv4[1],100-npv4[2],100-npv4[3],100-npv4[4]]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))

# Set positions for the bars
pos = np.arange(len(methods))

# Plot bars for each dataset
bars1 = ax.bar(pos - 1.5*width, for1, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, for2, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, for3, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, for4, width=width, label='Emotions data', color='#FF5768')

ax.set_ylabel('FOR(%)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)
legend = ax.legend(prop=legend_font,loc='lower right')
ax.tick_params(axis='y', labelsize=16)

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  # Rotate and anchor at bottom
                    # You can adjust rotation and alignment as needed
ax.set_ylim(0, 30)
plt.tight_layout()
plt.show()



# FPR
csfont = {'fontname':'Times New Roman'}
methods = ['NasNet\nXGB','DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
fpr1 = [100-spe1[0],100-spe1[1],100-spe1[2],100-spe1[3],100-spe1[4]]
fpr2 = [100-spe2[0],100-spe2[1],100-spe2[2],100-spe2[3],100-spe2[4]]
fpr3 = [100-spe3[0],100-spe3[1],100-spe3[2],100-spe3[3],100-spe3[4]]
fpr4 = [100-spe4[0],100-spe4[1],100-spe4[2],100-spe4[3],100-spe4[4]]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))

# Set positions for the bars
pos = np.arange(len(methods))

# Plot bars for each dataset
bars1 = ax.bar(pos - 1.5*width, fpr1, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, fpr2, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, fpr3, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, fpr4, width=width, label='Emotions data', color='#FF5768')

ax.set_ylabel('FPR(%)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)
legend = ax.legend(prop=legend_font,loc='lower right')
ax.tick_params(axis='y', labelsize=16)

# Adding labels for each bar
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  # Rotate and anchor at bottom
ax.set_ylim(0, 30)
plt.tight_layout()
plt.show()

# fnr
csfont = {'fontname':'Times New Roman'}
methods = ['NasNet\nXGB','DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
fnr1 = [100-Recall1[0],100-Recall1[1],100-Recall1[2],100-Recall1[3],100-Recall1[4]]
fnr2 = [100-Recall2[0],100-Recall2[1],100-Recall2[2],100-Recall2[3],100-Recall2[4]]
fnr3 = [100-Recall3[0],100-Recall3[1],100-Recall3[2],100-Recall3[3],100-Recall3[4]]
fnr4 = [100-Recall4[0],100-Recall4[1],100-Recall4[2],100-Recall4[3],100-Recall4[4]]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))

# Set positions for the bars
pos = np.arange(len(methods))

# Plot bars for each dataset
bars1 = ax.bar(pos - 1.5*width, fnr1, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, fnr2, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, fnr3, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, fnr4, width=width, label='Emotions data', color='#FF5768')

ax.set_ylabel('FNR(%)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)
legend = ax.legend(prop=legend_font,loc='lower right')
ax.tick_params(axis='y', labelsize=16)

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  # Rotate and anchor at bottom
                    # You can adjust rotation and alignment as needed

ax.set_ylim(0, 30)
plt.tight_layout()
plt.show()


# FDR
csfont = {'fontname':'Times New Roman'}
methods = ['NasNet\nXGB','DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
FDR1 = [100-Pre1[0],100-Pre1[1],100-Pre1[2],100-Pre1[3],100-Pre1[4]]
FDR2 = [100-Pre2[0],100-Pre2[1],100-Pre2[2],100-Pre2[3],100-Pre2[4]]
FDR3 = [100-Pre3[0],100-Pre3[1],100-Pre3[2],100-Pre3[3],100-Pre3[4]]
FDR4 = [100-Pre4[0],100-Pre4[1],100-Pre4[2],100-Pre4[3],100-Pre4[4]]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))

# Set positions for the bars
pos = np.arange(len(methods))

# Plot bars for each dataset
bars1 = ax.bar(pos - 1.5*width, FDR1, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, FDR2, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, FDR3, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, FDR4, width=width, label='Emotions data', color='#FF5768')

ax.set_ylabel('FDR(%)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)
legend = ax.legend(prop=legend_font,loc='lower right')
ax.tick_params(axis='y', labelsize=16)

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  # Rotate and anchor at bottom
                    # You can adjust rotation and alignment as needed

ax.set_ylim(0, 40)
plt.tight_layout()
plt.show()


# f1
csfont = {'fontname':'Times New Roman'}
methods = ['NasNet\nXGB','DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
f11 = [2*(Pre1[0]*Recall1[0])/(Pre1[0]+Recall1[0]),2*(Pre1[1]*Recall1[1])/(Pre1[1]+Recall1[1]),2*(Pre1[2]*Recall1[2])/(Pre1[2]+Recall1[2]),2*(Pre1[3]*Recall1[3])/(Pre1[3]+Recall1[3]),2*(Pre1[4]*Recall1[4])/(Pre1[4]+Recall1[4])]
f12 = [2*(Pre2[0]*Recall2[0])/(Pre2[0]+Recall2[0]),2*(Pre2[1]*Recall2[1])/(Pre2[1]+Recall2[1]),2*(Pre2[2]*Recall2[2])/(Pre2[2]+Recall2[2]),2*(Pre2[3]*Recall2[3])/(Pre2[3]+Recall2[3]),2*(Pre2[4]*Recall2[4])/(Pre2[4]+Recall2[4])]
f13 = [2*(Pre3[0]*Recall3[0])/(Pre3[0]+Recall3[0]),2*(Pre3[1]*Recall3[1])/(Pre3[1]+Recall3[1]),2*(Pre3[2]*Recall3[2])/(Pre3[2]+Recall3[2]),2*(Pre3[3]*Recall3[3])/(Pre3[3]+Recall3[3]),2*(Pre3[4]*Recall3[4])/(Pre3[4]+Recall3[4])]
f14 = [2*(Pre4[0]*Recall4[0])/(Pre4[0]+Recall4[0]),2*(Pre4[1]*Recall4[1])/(Pre4[1]+Recall4[1]),2*(Pre4[2]*Recall4[2])/(Pre4[2]+Recall4[2]),2*(Pre4[3]*Recall4[3])/(Pre4[3]+Recall4[3]),2*(Pre4[4]*Recall3[4])/(Pre4[4]+Recall4[4])]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))

# Set positions for the bars
pos = np.arange(len(methods))

# Plot bars for each dataset
bars1 = ax.bar(pos - 1.5*width, f11, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, f12, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, f13, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, f14, width=width, label='Emotions data', color='#FF5768')

ax.set_ylabel('F1_Score (%)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)
legend = ax.legend(prop=legend_font,loc='lower right')
ax.tick_params(axis='y', labelsize=16)

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  # Rotate and anchor at bottom
                    # You can adjust rotation and alignment as needed

ax.set_ylim(0, 110)
plt.tight_layout()
plt.show()


# error
csfont = {'fontname':'Times New Roman'}
methods = ['NasNet\nXGB','DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
error1 = [100-Acc1[0],100-Acc1[1],100-Acc1[2],100-Acc1[3],100-Acc1[4]]
error2 = [100-Acc2[0],100-Acc2[1],100-Acc2[2],100-Acc2[3],100-Acc2[4]]
error3 = [100-Acc3[0],100-Acc3[1],100-Acc3[2],100-Acc3[3],100-Acc3[4]]
error4 = [100-Acc4[0],100-Acc4[1],100-Acc4[2],100-Acc4[3],100-Acc4[4]]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))

# Set positions for the bars
pos = np.arange(len(methods))

# Plot bars for each dataset
bars1 = ax.bar(pos - 1.5*width, error1, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, error2, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, error3, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, error4, width=width, label='Emotions data', color='#FF5768')

ax.set_ylabel('Error(%)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)
legend = ax.legend(prop=legend_font,loc='lower right')
ax.tick_params(axis='y', labelsize=16)
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  # Rotate and anchor at bottom
                    # You can adjust rotation and alignment as needed

ax.set_ylim(0, 35)
plt.tight_layout()
plt.show()


# jaccard_similarity
csfont = {'fontname':'Times New Roman'}
methods = ['NasNet\nXGB','DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
JS1 = [0.872*100,0.813*100,0.776*100,0.722*100,0.757*100]
JS2 = [0.856*100,0.834*100,0.725*100,0.78*100,0.674*100]
JS3 = [0.891*100,0.828*100,0.708*100,0.754*100,0.69*100]
JS4 = [0.852*100,0.802*100,0.829*100,0.765*100,0.712*100]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))
pos = np.arange(len(methods))
bars1 = ax.bar(pos - 1.5*width, JS1, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, JS2, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, JS3, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, JS4, width=width, label='Emotions data', color='#FF5768')
ax.set_ylabel('Jaccard_similarity(%)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)
legend = ax.legend(prop=legend_font,loc='lower right')
ax.tick_params(axis='y', labelsize=16)
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  
                    
ax.set_ylim(0, 110)
plt.tight_layout()
plt.show()


# MCC
csfont = {'fontname':'Times New Roman'}
methods = ['NasNet\nXGB','DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
mcc1 = [0.984*100,0.927*100,0.87*100,0.81*100,0.844*100]
mcc2 = [0.885*100,0.816*100,0.848*100,0.769*100,0.721*100]
mcc3 = [0.911*100,0.893*100,0.822*100,0.733*100,0.688*100]
mcc4 = [0.82*100,0.761*100,0.727*100,0.653*100,0.638*100]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))
pos = np.arange(len(methods))
bars1 = ax.bar(pos - 1.5*width, mcc1, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, mcc2, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, mcc3, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, mcc4, width=width, label='Emotions data', color='#FF5768')
ax.set_ylabel('MCC (%)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)
legend = ax.legend(prop=legend_font,loc='lower right')
ax.tick_params(axis='y', labelsize=16)
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  
                    
ax.set_ylim(0, 110)
plt.tight_layout()
plt.show()


#Kappa
csfont = {'fontname':'Times New Roman'}
methods = ['NasNet\nXGB','DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
Kappa1 = [0.972*100,0.906*100,0.858*100,0.834*100,0.792*100]
Kappa2 = [0.937*100,0.854*100,0.826*100,0.738*100,0.758*100]
Kappa3 = [0.875*100,0.832*100,0.764*100,0.75*100,0.77*100]
Kappa4 = [0.89*100,0.861*100,0.802*100,0.792*100,0.763*100]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))
pos = np.arange(len(methods))
bars1 = ax.bar(pos - 1.5*width, Kappa1, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, Kappa2, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, Kappa3, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, Kappa4, width=width, label='Emotions data', color='#FF5768')
ax.set_ylabel('Kappa (%)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)
legend = ax.legend(prop=legend_font,loc='lower right')
ax.tick_params(axis='y', labelsize=16)
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  
                    
ax.set_ylim(0, 110)
plt.tight_layout()
plt.show()


#MK
csfont = {'fontname':'Times New Roman'}
methods = ['NasNet\nXGB','DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
MK1 = [0.893*100,0.885*100,0.837*100,0.777*100,0.729*100]
MK2 = [0.926*100,0.815*100,0.861*100,0.855*100,0.7708*100]
MK3 = [0.938*100,0.827*100,0.792*100,0.815*100,0.798*100]
MK4 = [0.91*100,0.8071*100,0.844*100,0.783*100,0.732*100]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))
pos = np.arange(len(methods))
bars1 = ax.bar(pos - 1.5*width, MK1, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, MK2, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, MK3, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, MK4, width=width, label='Emotions data', color='#FF5768')
ax.set_ylabel('MK (%)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)
legend = ax.legend(prop=legend_font,loc='lower right')
ax.tick_params(axis='y', labelsize=16)
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  
                    
ax.set_ylim(0, 110)
plt.tight_layout()
plt.show()

#FM
csfont = {'fontname':'Times New Roman'}
methods = ['NasNet\nXGB','DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
FM1 = [0.911*100,0.879*100,0.804*100,0.849*100,0.751*100]
FM2 = [0.928*100,0.904*100,0.895*100,0.837*100,0.886*100]
FM3 = [0.960*100,0.816*100,0.792*100,0.871*100,0.824*100]
FM4 = [0.935*100,0.913*100,0.756*100,0.760*100,0.658*100]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))
pos = np.arange(len(methods))
bars1 = ax.bar(pos - 1.5*width, FM1, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, FM2, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, FM3, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, FM4, width=width, label='Emotions data', color='#FF5768')
ax.set_ylabel('FM (%)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)
legend = ax.legend(prop=legend_font,loc='lower right')
ax.tick_params(axis='y', labelsize=16)
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  
                    
ax.set_ylim(0, 110)
plt.tight_layout()
plt.show()

#hamming_loss
csfont = {'fontname':'Times New Roman'}
methods = ['NasNet\nXGB','DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
HL1 = [100-FM1[0],100-FM1[1],100-FM1[2],100-FM1[3],100-FM1[4]]
HL2 = [100-FM2[0],100-FM2[1],100-FM2[2],100-FM2[3],100-FM2[4]]
HL3 = [100-FM3[0],100-FM3[1],100-FM3[2],100-FM3[3],100-FM3[4]]
HL4 = [100-FM4[0],100-FM4[1],100-FM4[2],100-FM4[3],100-FM4[4]]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))
pos = np.arange(len(methods))
bars1 = ax.bar(pos - 1.5*width, HL1, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, HL2, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, HL3, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, HL4, width=width, label='Emotions data', color='#FF5768')
ax.set_ylabel('Hamming loss (%)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)
legend = ax.legend(prop=legend_font,loc='upper left')
ax.tick_params(axis='y', labelsize=16)
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  
                    
# ax.set_ylim(0, 110)
plt.tight_layout()
plt.show()


#Training Time(sec)
csfont = {'fontname':'Times New Roman'}
methods = ['NasNet\nXGB','DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
TR1 = [1.088,1.24,1.57,1.88,1.98]
TR2 = [0.109,0.126,0.145,0.172,0.179]
TR3 = [0.71,0.88,1.25,1.38,1.74]
TR4 = [0.38,0.46,0.73,1.11,1.43]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))
pos = np.arange(len(methods))
bars1 = ax.bar(pos - 1.5*width, TR1, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, TR2, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, TR3, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, TR4, width=width, label='Emotions data', color='#FF5768')
ax.set_ylabel('Training Time(sec)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)
legend = ax.legend(prop=legend_font,loc='lower right')
ax.tick_params(axis='y', labelsize=16)
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  
                    
# ax.set_ylim(0, 110)
plt.tight_layout()
plt.show()


#Testing Time(sec)
csfont = {'fontname':'Times New Roman'}
methods = ['NasNet\nXGB','DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
TS1 = [0.0039,0.0054,0.0083,0.0136,0.0154]
TS2 = [0.015,0.026,0.057,0.075,0.088]
TS3 = [0.019,0.026,0.038,0.069,0.092]
TS4 = [0.0042,0.0083,0.0142,0.0176,0.0185]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))
pos = np.arange(len(methods))
bars1 = ax.bar(pos - 1.5*width, TS1, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, TS2, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, TS3, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, TS4, width=width, label='Emotions data', color='#FF5768')
ax.set_ylabel('Testing Time(sec)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)
legend = ax.legend(prop=legend_font,loc='upper left')
ax.tick_params(axis='y', labelsize=16)
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  
                    
# ax.set_ylim(0, 110)
plt.tight_layout()
plt.show()


#Execution Time(sec)
csfont = {'fontname':'Times New Roman'}
methods = ['NasNet\nXGB','DenseNet\nSVM', 'AlexNet\nKNN', 'VGGNet\nRF', 'ResNet\nANN']
EX1 = [TR1[0]+TS1[0],TR1[1]+TS1[1],TR1[2]+TS1[2],TR1[3]+TS1[3],TR1[4]+TS1[4]]
EX2 = [TR2[0]+TS2[0],TR2[1]+TS2[1],TR2[2]+TS2[2],TR2[3]+TS2[3],TR2[4]+TS2[4]]
EX3 = [TR3[0]+TS3[0],TR3[1]+TS3[1],TR3[2]+TS3[2],TR3[3]+TS3[3],TR3[4]+TS3[4]]
EX4 = [TR4[0]+TS4[0],TR4[1]+TS4[1],TR4[2]+TS4[2],TR4[3]+TS4[3],TR4[4]+TS4[4]]
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))
pos = np.arange(len(methods))
bars1 = ax.bar(pos - 1.5*width, EX1, width=width, label='Epileptic seizures data', color='#00A5E3')
bars2 = ax.bar(pos - 0.5*width, EX2, width=width, label='Stress Detection data', color='#8DD7BF')
bars3 = ax.bar(pos + 0.5*width, EX3, width=width, label='Stroke data', color='#FF96C5')
bars4 = ax.bar(pos + 1.5*width, EX4, width=width, label='Emotions data', color='#FF5768')
ax.set_ylabel('Execution Time(sec)', fontsize=22, **csfont)
ax.set_xticks(pos)
ax.set_xticklabels(methods, fontname="Times New Roman", fontsize=22)
legend = ax.legend(prop=legend_font,loc='lower right')
ax.tick_params(axis='y', labelsize=16)
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(6, 20), textcoords='offset points', ha='center', va='bottom', fontsize=15,
                    rotation=90, rotation_mode='anchor')  
                    
# ax.set_ylim(0, 110)
plt.tight_layout()
plt.show()









