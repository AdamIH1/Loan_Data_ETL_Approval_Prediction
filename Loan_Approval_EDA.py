import pandas as pd 

df = pd.read_csv('Loan Data Set')

df = df.drop(columns=['Loan_ID']) ## Dropping Loan ID

df.info()

df.head()



categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                       'Property_Area','Credit_History','Loan_Amount_Term']

numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']


import seaborn as sns
import matplotlib.pyplot as plt


fig,axes = plt.subplots(4,2,figsize=(13,15))
for idx,cat_col in enumerate(categorical_columns):
    row,col = idx//2,idx%2
    sns.countplot(x=cat_col,data=df,hue='Loan_Status',ax=axes[row,col])


plt.subplots_adjust(hspace=1)

