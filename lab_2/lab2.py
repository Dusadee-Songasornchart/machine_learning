import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

data = pd.read_csv("https://raw.githubusercontent.com/srivatsan88/YouTubeLI/master/dataset/churn_data_st.csv")

datacon = pd.DataFrame({'tenure':data['tenure'],'ServiceCount':data['ServiceCount'],'MonthlyCharges':data['MonthlyCharges'],'TotalCharges':data['TotalCharges']})


#removecustomerID
data.drop(data.columns[[0]], axis=1, inplace=True)

#print(data)
#print(data.dtypes)
print(datacon)

#corr
#print(datacon['tenure'].corr(data['ServiceCount']))
#print(datacon['MonthlyCharges'].corr(data['TotalCharges']))
ax = sns.heatmap(datacon)
plt.show()