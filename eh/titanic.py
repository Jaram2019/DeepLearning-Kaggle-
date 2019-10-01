
import numpy as np
import tensorflow as tf
import pandas as pd

# 최대 줄 수 설정
pd.set_option('display.max_rows', 500)
# 최대 열 수 설정
pd.set_option('display.max_columns', 500)
# 표시할 가로의 길이
pd.set_option('display.width', 1000)

train=pd.read_csv('train.csv',delimiter=',')
test=pd.read_csv('test.csv',delimiter=',')
print(train)
train['Title'] =train['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)
test['Title']=train['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)
#print(train['Title'].value_counts())
title_mapping={"Mr" : 0, "Miss" : 1, "Mrs" : 2}
train['Title']=train['Title'].map(title_mapping)
train['Title'].fillna(3,inplace=True)
test['Title']=test['Title'].map(title_mapping)
test['Title'].fillna(3,inplace=True)
#print(train)
#print(test)
#print(train.groupby('Title').sum())
train.drop('Name',axis=1,inplace=True)
test.drop('Name',axis=1,inplace=True)
#print(train)

sex_mapping={'male':0,'female':1}
train['Sex']=train['Sex'].map(sex_mapping)
test['Sex']=test['Sex'].map(sex_mapping)
#print(train.groupby('Sex').sum()['Survived'])

train['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)
train['Age']=train['Age'].apply(lambda x : int(x))
test['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)
test['Age']=test['Age'].apply(lambda x : int(x))
train.loc[(train['Age']<=20), 'Age']=0
train.loc[(train['Age']>20) & (train['Age']<=40),'Age']=1
train.loc[(train['Age']>40) & (train['Age']<=60),'Age']=2
train.loc[(train['Age']>60),'Age']=3
test.loc[test['Age']<=20, 'Age']=0
test.loc[(test['Age']>20) & (test['Age']<=40), 'Age']=1
test.loc[(test['Age']>40) & (test['Age']<=60), 'Age']=2
test.loc[(test['Age']>60), 'Age']=3

#print(train.Cabin.value_counts())
em_mapping={'S':0,'C':1,'Q':2}
train['Embarked']=train['Embarked'].map(em_mapping)
test['Embarked']=test['Embarked'].map(em_mapping)

train['Cabin']=train['Cabin'].str[:1]
test['Cabin']=test['Cabin'].str[:1]
cabin_map={'A' : 0, 'B' : 0.4,'c':0.8,'D':1.2,'E':1.6,'F':2,'G':2.4,'T':2.8}
train['Cabin']=train['Cabin'].map(cabin_map)
test['Cabin']=test['Cabin'].map(cabin_map)
train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'), inplace=True)
test['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'), inplace=True)


family_map={1:0,2:1,3:2,4:3}
train['SibSp']=train['SibSp'].map(family_map)
test.loc[test['SibSp']>4,'SibSp']=3
test['SibSp']=test['SibSp'].map(family_map)
train['SibSp'].fillna(train.groupby('Pclass')['SibSp'].transform('median'),inplace=True)
test['SibSp'].fillna(test.groupby('Pclass')['SibSp'].transform('median'),inplace=True)

train.drop(['Parch','Ticket','Fare'],axis=1,inplace=True)
test.drop(['Parch','Ticket','Fare'],axis=1,inplace=True)
print(train)
train.to_csv('post_train.csv',mode='w')
test.to_csv('post_test.csv',mode='w')
#--------------
#temp1=train.groupby('SibSp').count()['Survived']
#temp2=train.groupby('SibSp').sum()['Survived']
#print((temp1-temp2)/temp1)
