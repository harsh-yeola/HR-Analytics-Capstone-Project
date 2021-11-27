# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 11:11:14 2021

@author: Dell
"""

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

os.getcwd()
os.chdir('E:\DataSkills 2021\Imarticus DSP\Capstone Project\HR Analytics')

# 1

hr_16_17 = pd.read_excel('staff utlz latest 16-17_masked.xlsx',header=[0,1]).reset_index()

hr_16_17.columns = ['-'.join(col).strip() for col in hr_16_17.columns.values]
hr_16_17.drop('index-',axis=1,inplace=True)

col_list = [a.split('-')[-1] for a in hr_16_17.columns]
col_list[0]='Employee No'

for i in range(115):
    hr_16_17.rename(columns={hr_16_17.columns[i]:col_list[i]},inplace=True)

hr_16_17_1 = hr_16_17.iloc[:,107:115]
hr_16_17_2 = hr_16_17.iloc[:,0:11]

hr_16_17_total = pd.concat([hr_16_17_2,hr_16_17_1],axis=1)

hr_16_17_total.loc[hr_16_17_total['Termination Date']=='-','Termination Date'] = pd.to_datetime('today')
hr_16_17_total[['Join Date','Termination Date']]=hr_16_17_total[['Join Date','Termination Date']].apply(pd.to_datetime)
hr_16_17_total[['Join Date','Termination Date']]=hr_16_17_total[['Join Date','Termination Date']].apply(lambda t: t.dt.floor('d'))

hr_16_17_total['Tenure']=hr_16_17_total['Termination Date']-hr_16_17_total['Join Date']
hr_16_17_total['Tenure']=hr_16_17_total['Tenure'].dt.days



# 2

hr_17_18 = pd.read_excel('staff utlz latest 17-18_masked.xlsx',header=[0,1]).reset_index()

hr_17_18_1 = hr_17_18.iloc[:,108:116]
hr_17_18_2 = hr_17_18.iloc[:,0:12]
hr_17_18_total = pd.concat([hr_17_18_2,hr_17_18_1],axis=1)
#[hr_17_18.columns[i][0].strftime('%b-%Y') for i in range(12,108)]

hr_17_18_total.columns = ['-'.join(col).strip() for col in hr_17_18_total.columns.values]
hr_17_18_total.drop('index-',axis=1,inplace=True)

col_list = [a.split('-')[-1] for a in hr_17_18_total.columns]
# col_list[0]='Employee No'

for i in range(19):
    hr_17_18_total.rename(columns={hr_17_18_total.columns[i]:col_list[i]},inplace=True)

#hr_17_18_1 = hr_17_18.iloc[:,107:115]
#hr_17_18_2 = hr_17_18.iloc[:,0:11]

# hr_17_18_total = pd.concat([hr_hr_17_18_2,hr_hr_17_18_1],axis=1)

hr_17_18_total['Termination Date'].fillna(value=pd.to_datetime('today'),inplace=True)
hr_17_18_total[['Join Date','Termination Date']]=hr_17_18_total[['Join Date','Termination Date']].apply(pd.to_datetime)
hr_17_18_total[['Join Date','Termination Date']]=hr_17_18_total[['Join Date','Termination Date']].apply(lambda t: t.dt.floor('d'))

hr_17_18_total['Tenure']=hr_17_18_total['Termination Date']-hr_17_18_total['Join Date']
hr_17_18_total['Tenure']=hr_17_18_total['Tenure'].dt.days


# 3

hr_total = pd.concat([hr_16_17_total,hr_17_18_total],axis=0,ignore_index=True)

hr_total.loc[hr_total['Utilization%']=='-','Utilization%']=0
hr_total['Utilization%']=pd.to_numeric(hr_total['Utilization%'])

hr_total.to_csv('hr_total.csv',index=False)

hr_total_category = hr_total[['Profit Center', 'Employee Position','Employee Location','Employee Category']]
                 
hr_total_numerical = hr_total[['Total Hours', 'Total Available Hours', 'Work Hours', 'Leave Hours',
                       'Training Hours', 'BD Hours', 'NC Hours', 'Utilization%', 'Tenure']]     

hr_total_status = hr_total[['Employee No','Current Status']]

hr_total_status['Current Status'].value_counts()
hr_total_status.loc[hr_total_status['Current Status']!='Resigned','Current Status']='Active'

hr_total_category = pd.get_dummies(hr_total_category,drop_first=True)


hr_final=pd.concat([hr_total_category,hr_total_numerical,hr_total_status['Current Status']],axis=1)

# 4

X=hr_final.iloc[:,:-1]
y=hr_final.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)

hr_predict=clf.predict(X_test)
hr_predict_df=pd.DataFrame({'Test Data':y_test,'Prediction':hr_predict})

cm=confusion_matrix(y_test, hr_predict)
cm

print({'tn':cm[0,0],'fp':cm[0,1],'fn':cm[1,0],'tp':cm[1,1]})



