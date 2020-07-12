import numpy as np
import pandas as pd
import scipy
import sklearn.model_selection as ms
import csv
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics


# disable warnings
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# print('x' in np.arange(5))   #returns False, without Warning

# parse csv as dataframes
def task_2a():
    
    # initialise 2D array for csv 
    task2a_csv_array = [['feature', 'median', 'mean', 'variance']]
    
    # read csv files as dataframe
    life_df = pd.read_csv("life.csv")
    world_df = pd.read_csv("world.csv")
   
    # merge dataframes on common columns (country and country code)
    world_df = world_df.rename(columns={'Country Name': 'Country', 'Time':'Year'})
    new_df = pd.merge(life_df, world_df,  how='inner', on=['Country','Country Code'])
    
    # remove rows with null 'life expectancy' values
   
    new_df = new_df.dropna(axis=0, subset=['Life expectancy at birth (years)'])
    
    # split into training and test sets with random state of 100
    X = new_df.iloc[:,5:] # learn from these data
    y = new_df.loc[:,'Life expectancy at birth (years)'] # expected results
    # X_train = input data used to train the model
    # y_train = output data used to train the model
    # X_test = input data used to test the model
    # y_test = output data used to test the model 
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, train_size=2/3, test_size = 1/3, random_state=100)
        
    # turn strings from X_train and X_test to NaN (inputs)
    for column in X_train.columns:
        X_train[column] = pd.to_numeric(X_train[column], errors='coerce')
        
    for column in X_test.columns:
        X_test[column] = pd.to_numeric(X_test[column], errors='coerce')
        
    # fill the NaN values in X_test and X_train with median of X_train

    for col in X_train.select_dtypes(include=np.number): 
        X_train[col] = X_train[col].fillna(X_train[col].median())
    
    for col in X_test.select_dtypes(include=np.number): 
        X_test[col] = X_test[col].fillna(X_train[col].median())    
    
    # append feature, median, mean, variance to array for csv
    for feature, values in X_train.iteritems():
        curr_row = []
        curr_row.append(feature)
        curr_row.append('{:.{width}f}'.format(values[:].median(), width=3))
        curr_row.append('{:.{width}f}'.format(values[:].mean(), width=3))
        curr_row.append('{:.{width}f}'.format(values[:].var(), width=3))
        task2a_csv_array.append(curr_row)
        
    # scale training set and test set
    scaler = preprocessing.StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test) # DO NOT FIT TO TEST because already fit to train
    
    # building decision tree model
    dt_classifier = tree.DecisionTreeClassifier(max_depth=4)
    dt_classifier.fit(scaled_X_train, y_train)
    decision_tree_accu = dt_classifier.score(scaled_X_test, y_test)
    print('Accuracy of decision tree: {:.{width}f}'.format(decision_tree_accu, width=3))
    
    # building kNN models
    k5_classifier = KNeighborsClassifier(n_neighbors=5) 
    k10_classifier = KNeighborsClassifier(n_neighbors=10)
    
    k5_classifier.fit(scaled_X_train, y_train)
    k10_classifier.fit(scaled_X_train, y_train)
    
    # evaluating accuracy method 1 (~0.75 for both)
    k5_test_accu = k5_classifier.score(scaled_X_test, y_test)
    k10_test_accu = k10_classifier.score(scaled_X_test, y_test)
    print('Accuracy of k-nn (k=5): {:.{width}f}'.format(k5_test_accu, width=3))
    print('Accuracy of k-nn (k=10): {:.{width}f}'.format(k10_test_accu, width=3))
    
    # write to csv
    with open("task2a.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(task2a_csv_array)
    
task_2a()
