from re import X
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

def remove_outlier(df_in,col_name_x,col_name_y):
    q1_x = df_in[col_name_x].quantile(0.25)
    q3_x = df_in[col_name_x].quantile(0.75)
    iqr_x = q3_x-q1_x #Interquartile range
    fence_low_x  = q1_x-1.5*iqr_x
    fence_high_x = q3_x+1.5*iqr_x
    q1_y = df_in[col_name_y].quantile(0.25)
    q3_y = df_in[col_name_y].quantile(0.75)
    iqr_y = q3_y-q1_y #Interquartile range
    fence_low_y  = q1_y-1.5*iqr_y
    fence_high_y = q3_y+1.5*iqr_y
    df_out = df_in.loc[((df_in[col_name_x] > fence_low_x) & (df_in[col_name_x] < fence_high_x)) & ((df_in[col_name_y] > fence_low_y) & (df_in[col_name_y] < fence_high_y))]
    return df_out

scale = preprocessing.MinMaxScaler()

data = pd.read_csv("lab_1/Data_example.csv")

##replace form stackoverflow
data.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
data['X'].fillna(data['X'],inplace=True)
data['Y'].fillna(value = data['Y'],inplace=True)
data['Z'].fillna(np.nan,inplace=True)



#convertdata

data['X'] = pd.to_numeric(data['X'], errors='coerce')
data['Y'] = pd.to_numeric(data['Y'], errors='coerce')
data['X'] = data['X'].astype('Int64')
data['Y'] = data['Y'].astype('float64')

##duplicates
data = data.drop_duplicates()

#drop na
data = data[data.isnull().sum(axis=1)<2]

#replace na
Xmedian = data['X'].median()
Xmedian = Xmedian.astype(int)
Ymean = data['Y'].mean()

data['X'].fillna(Xmedian , inplace=True)
data['Y'].fillna(Ymean, inplace=True)
data['Z'].ffill(inplace=True)

##print data
#print(data)
##describe
#print(data.describe())
#print(data.dtypes)
#print(data.shape)

#remove outliner
data = remove_outlier(data,'X','Y')



#tranformdata min max scalr

data[["X","Y"]] = pd.DataFrame(scale.fit_transform(data[["X","Y"]].values), columns=["X","Y"], index=data.index)



#printboxplot
fig, ax = plt.subplots(1,2)
ax[0].boxplot(data['X'])
ax[0].set_xticklabels('X')
ax[1].boxplot(data['Y'])
ax[1].set_xticklabels('Y')

#lableencoder
le = preprocessing.LabelEncoder()
le.fit(data['Z'])
data['Z_category'] = le.transform(data['Z'])

#onehotencoder
enc = preprocessing.OneHotEncoder()
datae = enc.fit_transform(data['Z'].values.reshape(-1,1)).toarray()
dataez = pd.DataFrame(datae,columns=['bird','cat','dog'])




#reset drop index

data = data.reset_index(drop=True)
data = data.join(dataez)


print(data)
#plt.show()
