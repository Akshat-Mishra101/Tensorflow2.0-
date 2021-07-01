import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


dftrain =pd.read_csv("Datasets/train.csv")
dfeval=pd.read_csv("Datasets/eval.csv")

#print(dftrain.loc[0])#prints the first entry of the csv

y_train=dftrain.pop('survived')
y_eval=dfeval.pop('survived')

categorical_columns=['sex','n_siblings_spouses','parch','class','deck','embark_town','alone']#discrete values
numeric_columns=['age','fare']#numeric values

feature_columns=[]
for feature in categorical_columns:
    vocabulary =dftrain[feature].unique()# this will iteratre  through the different features and store all the unique values of the columns in a list
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature,vocabulary))#creates a numpy array with the given vocabulary for a given feature and adds it to our feature columns array


for feature_name in numeric_columns:
    feature_columns.append(tf.feature_column.numeric_column(feature,dtype=tf.float32))

print(feature_columns['sex'])

#data can be passed into the machine learning trainer as batches or as a whole
#preferable to send in batches when dataset is hugeee



#print(dftrain["age"])
#the first step in linear regression is always cleaning the database and creating the feature columns

