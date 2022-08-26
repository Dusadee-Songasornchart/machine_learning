import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

data = pd.read_csv("https://raw.githubusercontent.com/srivatsan88/YouTubeLI/master/dataset/churn_data_st.csv")


#removecustomerID
data.drop(data.columns[[0]], axis=1, inplace=True)


#fillna
data = data.replace(r'^\s+$', np.nan, regex=True)

datacon = pd.DataFrame({'tenure':data['tenure'],'ServiceCount':data['ServiceCount'],'MonthlyCharges':data['MonthlyCharges'],'TotalCharges':data['TotalCharges']})

#print(data)
#print(data.dtypes)


columns = ['tenure','ServiceCount', 'MonthlyCharges','TotalCharges']



#corr

ax = sns.heatmap(datacon.corr())
lower = pd.DataFrame(np.tril(datacon.corr(), -1),columns = datacon.corr().columns)
to_drop = [column for column in lower if any(lower[column] > 0.6)]
datacon.drop(to_drop, inplace=True, axis=1)
#print(datacon)
#print(datacon.describe())
#plt.show()

data_23 = pd.DataFrame({'Churn':data['Churn']})
le = preprocessing.LabelEncoder()
le.fit(data_23['Churn'])
data_23['Churn'] = le.transform(data_23['Churn'])

data_4 = pd.DataFrame({'gender':data['gender'],'Contract':data['Contract'],'PaperlessBilling':data['PaperlessBilling']})
#le.fit(data_4['gender'])
data_4['gender'] = le.fit_transform(data_4['gender'])
data_4['Contract'] = le.fit_transform(data_4['Contract'])

print(data_4)

#print(data)
#print(data_23)