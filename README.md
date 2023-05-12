# Feature-selection-techniques
# AIM
To Perform the various feature selection techniques on a dataset and save the data to a file.

# ALGORITHM
STEP 1

Read the given Data

STEP 2

Clean the Data Set using Data Cleaning Process

STEP 3

Apply Feature selection techniques to all the features of the data set

STEP 4

Save the data to the file

# CODE

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df=pd.read_csv('/content/titanic_dataset.csv')
df.head()

df.isnull()

df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
df.head()

df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()

plt.title("Dataset with outliers")
df.boxplot()
plt.show()

cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()

from sklearn.preprocessing import OrdinalEncoder
climate = ['C','S','Q']
en= OrdinalEncoder(categories = [climate])
df['Embarked']=en.fit_transform(df[["Embarked"]])
df.head()

from sklearn.preprocessing import OrdinalEncoder
climate = ['male','female']
en= OrdinalEncoder(categories = [climate])
df['Sex']=en.fit_transform(df[["Sex"]])
df.head()

from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])
df.head()

import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()
df1["Survived"]=np.sqrt(df["Survived"])
df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])
df1["Sex"]=np.sqrt(df["Sex"])
df1["Age"]=df["Age"]
df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])
df1["Fare"],parameters=stats.yeojohnson(df["Fare"])
df1["Embarked"]=df["Embarked"]
df1.skew()

import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1) 
y = df1["Survived"]       
<ipython-input-29-cc5f080780d2>:10: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only.
  X = df1.drop("Survived",1)

plt.figure(figsize=(7,6))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()

cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features

X_1 = sm.add_constant(X)
model = sm.OLS(y,X_1).fit()
model.pvalues

cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
['Pclass', 'Sex', 'Age', 'Fare']

model = LinearRegression()

rfe = RFE(model,step= 4)

X_rfe = rfe.fit_transform(X,y)  

model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)

nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,step=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
Optimum number of features: 1
Score with 1 features: 0.344046

cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, step=2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)
Index(['Pclass', 'Sex', 'SibSp'], dtype='object')

reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()
# OUTPUT
![image](https://github.com/varshini67t/Feature-selection-techniques/assets/107982953/67eee14b-9d3e-4a40-ae0b-b0c7112e4573)
![image](https://github.com/varshini67t/Feature-selection-techniques/assets/107982953/b3003fe3-46cf-47ba-9637-6830dab3c6ed)
![image](https://github.com/varshini67t/Feature-selection-techniques/assets/107982953/af1344bf-3edd-459c-b1ef-7053ed76bac9)
![image](https://github.com/varshini67t/Feature-selection-techniques/assets/107982953/0023cba8-3cb4-4bed-a32e-0cc9c587737a)
![image](https://github.com/varshini67t/Feature-selection-techniques/assets/107982953/7a3e6bc7-747e-4e3d-a15c-e3f046fc8a4d)
![image](https://github.com/varshini67t/Feature-selection-techniques/assets/107982953/3599fcc6-4c07-413a-90ad-896711e8ec0c)
![image](https://github.com/varshini67t/Feature-selection-techniques/assets/107982953/64a4ee51-8e54-4944-ad25-52c46fcb466a)
![image](https://github.com/varshini67t/Feature-selection-techniques/assets/107982953/fc9b6d90-acd1-479b-974f-d2d0d04b83ac)
![image](https://github.com/varshini67t/Feature-selection-techniques/assets/107982953/9483710c-2de3-49df-aee6-83f2969f5979)
![image](https://github.com/varshini67t/Feature-selection-techniques/assets/107982953/6acf1f23-c82e-415a-a127-acc35bc7078c)
![image](https://github.com/varshini67t/Feature-selection-techniques/assets/107982953/fd4f75ed-be40-489f-bf0f-b2bceaa049f7)
![image](https://github.com/varshini67t/Feature-selection-techniques/assets/107982953/ed03ea87-0e6a-4eef-bb1c-1f79f8ec9580)

  
# RESULT
Thus,we have executed various feature selection techniques.
