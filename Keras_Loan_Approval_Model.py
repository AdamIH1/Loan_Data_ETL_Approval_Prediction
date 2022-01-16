
"""
Created on Wed Nov  5 18:20:23 2021

@author: adamisaiahhansen

"""


# python imports 

import numpy as np
# np.random.seed(0) # for reproducibility
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from sklearn import preprocessing 
from sklearn.metrics import confusion_matrix, matthews_corrcoef


# uses pandas to import data

df = pd.read_csv('Loan_Approval_Data_Set.csv')

df = df.drop(columns=['Loan_ID']) # Dropping Loan ID

# preview data 

print('data shape',df.shape)

df.head()


# na / missing values 

df.isna().sum()

df = df.dropna()

df.isna().sum()

print('data shape',df.shape)

df.head()


# made variables for categorical and numerical columns 

categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 
                       'Self_Employed', 'Property_Area','Credit_History',
                                                       'Loan_Amount_Term']

numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']


# plotting categorical variables  

fig,axes = plt.subplots(4,2,figsize = (13,15))
for i,cat_col in enumerate(categorical_columns):
    row,col = i//2,i%2
    sns.countplot(x = cat_col,data = df,hue = 'Loan_Status',ax = axes[row,col])

plt.subplots_adjust(hspace = 1)


# plotting numerical variables 

fig,axes = plt.subplots(1,3,figsize = (17,5))

for i,cate in enumerate(numerical_columns):
    sns.boxplot(y = cate,data=df, x = 'Loan_Status', ax = axes[i])
      
print(df[numerical_columns].describe())



# data preprocesssing 

df_encoded = pd.get_dummies(df, drop_first = True)
df_encoded.head()
x = df_encoded.drop(columns = 'Loan_Status_Y')
y = df_encoded['Loan_Status_Y']


# splitting data into train and test 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, 
                                        stratify = y, random_state = 42)

# normalize data between 0 and 1 so weight in network have most effect 

normalize = preprocessing.MinMaxScaler()
x_train = normalize.fit_transform(x_train)
x_test = normalize.fit_transform(x_test)

# converted to np.array for keras
 
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# setup network 

num_epoch = 100

model = Sequential()
model.add(Dense(68, input_dim = 14, activation = 'relu'))
model.add(Dense(51, input_dim = 14, activation = 'relu'))
model.add(Dense(34, input_dim = 14, activation = 'relu'))
model.add(Dense(17, input_dim = 14, activation = 'relu'))
model.add(Dense(2,  input_dim = 14, activation = 'relu'))
model.add(Dense(17, input_dim = 14, activation = 'relu'))
model.add(Dense(34, input_dim = 14, activation = 'relu'))
model.add(Dense(51, input_dim = 14, activation = 'relu'))
model.add(Dense(68, input_dim = 14, activation = 'relu'))


model.add(Dense(1, activation='sigmoid'))


# fit the keras model on the dataset

model.compile(loss='binary_crossentropy', optimizer='adam', 
                                  metrics=['accuracy', 'mse'])
model.fit(x_train, y_train, epochs=num_epoch, batch_size = 1024)


# y pred

y_test_prediction = model.predict(x_test) 
y_test_prediction = np.round(y_test_prediction)


# MCC

MCC = matthews_corrcoef(y_test,y_test_prediction)
print(f'Matthews Correlation Coefficient: {MCC:.2f}')


# Confusion Matrix 

conf_matrix = confusion_matrix(y_test, y_test_prediction)
print('confusion matrix', conf_matrix)

fig, ax = plt.subplots(figsize = (7.5, 7.5))
ax.matshow(conf_matrix, cmap = plt.cm.Greens, alpha = 0.5)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x = j, y = i,s = conf_matrix[i, j], va = 'center', 
                                    ha = 'center', size = 'xx-large')
 
plt.xlabel('Predictions', fontsize = 18)
plt.ylabel('Actuals', fontsize = 18)
plt.title('Confusion Matrix', fontsize = 18)
plt.show()


