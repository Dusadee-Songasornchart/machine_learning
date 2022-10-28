import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn import metrics,model_selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import glob
from scipy import stats
import datetime as dt
from sklearn.neural_network import MLPClassifier
# Load data from csv 3 files
# acceleration.txt, heartrate.txt, labeled_sleep.txt
ACC = pd.read_csv("acceleration.txt", sep = ' ',names=['timedelta', 'accX', 'accY', 'accZ'])
HeartR = pd.read_csv("heartrate.txt", sep = ',',names=['timedelta', 'heartrate'])
SleepL = pd.read_csv("labeled_sleep.txt", sep = ' ',names=['timedelta', 'sleep'])

#check timedelta,min,max of acc,hr,slp
ACC_max_date = ACC["timedelta"].max()
ACC_min_date = ACC["timedelta"].min()

HR_max_date = HeartR["timedelta"].max()
HR_min_date = HeartR["timedelta"].min()

Slp_max_date = SleepL["timedelta"].max()
Slp_min_date = SleepL["timedelta"].min()

#จารให้หา start_timedelta, end_timedelta ส่วนตัวคิดว่าเป็น min กับ max ตามลำดับ
print("Acc Start : ",ACC_min_date," Acc End : ",ACC_max_date)
print("HeartR Start : ",HR_min_date," HeartR End : ",HR_max_date)
print("SleepL Start : ",Slp_min_date," SleepL End : ",Slp_max_date)


ACC_new = ACC[(ACC["timedelta"]> ACC_min_date) & (ACC["timedelta"] < ACC_max_date) & (ACC["timedelta"]> HR_min_date) & (ACC["timedelta"] < HR_max_date) &(ACC["timedelta"]> Slp_min_date) & (ACC["timedelta"] < Slp_max_date)]
HeartR_new = HeartR[(HeartR["timedelta"]> ACC_min_date) & (HeartR["timedelta"] < ACC_max_date) & (HeartR["timedelta"]> HR_min_date) & (HeartR["timedelta"] < HR_max_date) &(HeartR["timedelta"]> Slp_min_date) & (HeartR["timedelta"] < Slp_max_date)]
SleepL_new = SleepL[(SleepL["timedelta"]> ACC_min_date) & (SleepL["timedelta"] < ACC_max_date) & (SleepL["timedelta"]> HR_min_date) & (SleepL["timedelta"] < HR_max_date) &(SleepL["timedelta"]> Slp_min_date) & (SleepL["timedelta"] < Slp_max_date)]

# ACC_new = ACC[(ACC["timedelta"]> ACC_min_date) & (ACC["timedelta"] < ACC_max_date)]
# HeartR_new = HeartR[(HeartR["timedelta"]> HR_min_date) & (HeartR["timedelta"] < HR_max_date)]
# SleepL_new = SleepL[(SleepL["timedelta"]> Slp_min_date) & (SleepL["timedelta"] < Slp_max_date)]

# ------------ Rounding ACC (Rounding to 1 sec) -------------------------------
# Convert to datetime and round to second,
ACC_new['timedelta'] = pd.DataFrame(pd.to_timedelta(ACC_new['timedelta'], 'seconds').round('1s'))

# Average rounding duplicated time
df_acc_X = ACC_new.groupby('timedelta')['accX'].mean()
df_acc_Y = ACC_new.groupby('timedelta')['accY'].mean()
df_acc_Z = ACC_new.groupby('timedelta')['accZ'].mean()

ACC_new2 = pd.concat([df_acc_X, df_acc_Y, df_acc_Z], axis=1)
ACC_new2 = ACC_new2.reset_index()

ACC_new2['timedelta'] = ACC_new2['timedelta'] - ACC_new2['timedelta'].min()
print(ACC_new2)

# ------------ Rounding Heart Rate (Rounding to 1 sec) -------------------------------
HeartR_new['timedelta'] = pd.DataFrame(pd.to_timedelta(HeartR_new['timedelta'],"seconds").round('1s'))
# Resampling every 1s with median with ffill
resample_rule = '1s'
HeartR_new2 = HeartR_new.set_index('timedelta').resample(resample_rule,).median().ffill()
HeartR_new2 = HeartR_new2.reset_index()

HeartR_new2['timedelta'] = HeartR_new2['timedelta'] - HeartR_new2['timedelta'].min()
print(HeartR_new2)

# ------------ Rounding Sleep Label (Rounding to 1 sec) -------------------------------
SleepL_new['timedelta'] = pd.DataFrame(pd.to_timedelta(SleepL_new['timedelta'],"seconds").round('1s'))
# Resampling every 1s with median with ffill
resample_rule = '1s'
SleepL_new2 = SleepL_new.set_index('timedelta').resample(resample_rule,).median().ffill()
SleepL_new2 = SleepL_new2.reset_index()

SleepL_new2['timedelta'] = SleepL_new2['timedelta'] - SleepL_new2['timedelta'].min()
print(SleepL_new2)

# ------------Merge All Data -------------------------------
HeartR_new2['heartrate'].fillna(HeartR_new2.median())
SleepL_new2['sleep'].fillna(0)
df = []
df = pd.merge_asof(ACC_new2, HeartR_new2, on='timedelta')
df = pd.merge_asof(df, SleepL_new2, on = 'timedelta')
print(df)
# Drop column
df.drop(columns = ['timedelta'],inplace = True)
print(df)
# Standardized data
standard_scaler = StandardScaler()
feature_columns = ['accX', 'accY', 'accZ', 'heartrate']
label_columns = ['sleep']
df_feature = df[feature_columns] #standardized data of df_feature
df_feature = pd.DataFrame(standard_scaler.fit_transform(df_feature.values),index = df_feature.index,columns=df_feature.columns)
print(df_feature)
df_label = df[label_columns]
print(df_label)
# df_feature.plot()
# plt.show()
# df_label.plot()
# plt.show()

# ------------ 1D to 3D feature-------------------------------
# set sliding window parameter
slidingW = 100
Stride_step = 5
n_features = 4 #number of colums form df_feature
df_feature3D = np.array([],ndmin=2)
df_label_new = np.array([])

for t in range(0 , len(df_feature), Stride_step ):
    F3d = np.array(df_feature[t:t+slidingW],ndmin=2)
    # print(F3d[0])
    if len(F3d) <slidingW:
        break
    F3d.reshape(slidingW, n_features,1)
    # print(df_feature3D.shape)
    # print(F3d.shape)
    if df_feature3D.size == 0 :
        df_feature3D = F3d
    else:
        df_feature3D = np.dstack((df_feature3D,F3d))
    Labels = stats.mode(df_label[t : t+slidingW])
    # print(Labels)
    df_label_new = np.append(df_label_new,Labels[0])
    # print(df_feature3D.shape)
    # print(df_label_new.shape)
    # print(df_feature3D)

# df_feature3D = pd.DataFrame(df_feature3D) 
df_feature3D = np.swapaxes(df_feature3D,0,2)
print(df_feature3D.shape)
print(df_label_new.shape)
# ------------ Train-Test-Split 2D features -------------------------------
rseed=42
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(df_feature, df_label, test_size=0.3, random_state=rseed)

# ------------ Train-Test-Split 3D features -------------------------------
x3D_train, x3D_test, y3D_train, y3D_test = model_selection.train_test_split(df_feature3D, df_label_new, test_size=0.3, random_state=rseed)

# ------------ NN Architecture parameter -------------------------------
Hidden_Layer_param = (30, 30, 30)
mlp = MLPClassifier(hidden_layer_sizes = Hidden_Layer_param)
# View NN model parameters

# ------------ Training NN using 1D features -------------------------------
mlp.fit(X_train,Y_train)
mlp_pred = mlp.predict(X_test)
