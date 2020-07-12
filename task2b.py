import numpy as np
import pandas as pd
import scipy
import sklearn.model_selection as ms
import csv
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier


DIVIDER = "=========="

# disable warnings
pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=FutureWarning)
# print('x' in np.arange(5))   #returns False, without Warning


def task_2b():
    # read csv files as dataframe
    life_df = pd.read_csv("life.csv")
    world_df = pd.read_csv("world.csv")
    world_df = world_df.rename(columns={'Country Name': 'Country', 'Time':'Year'})
    
    # PREPROCESSING - from Question 2A
    # merge dataframes on common columns (country and country code)
    world_df = world_df.rename(columns={'Country Name': 'Country', 'Time':'Year'})
    new_df = pd.merge(life_df, world_df,  how='inner', on=['Country','Country Code'])
    # remove rows with null 'life expectancy' values
    new_df = new_df.dropna(axis=0, subset=['Life expectancy at birth (years)'])
    # split into training and test sets with random state of 100
    X1 = new_df.iloc[:,5:] # learn from these data
    X2 = new_df.iloc[:,1]
    # only keep Country Code with 20 original features, create a pointer to this df for reference to country split later
    X = pd.concat([X2.reset_index(drop=True), X1.reset_index(drop=True)], axis=1)
    y = new_df.loc[:,'Life expectancy at birth (years)'] # expected results
    X_train_with_country, X_test_with_country, y_train, y_test = ms.train_test_split(X, y, train_size=2/3, test_size = 1/3, random_state=100) 
    X_train_with_country_df = pd.DataFrame(X_train_with_country, index=X_train_with_country.index, columns=X_train_with_country.columns)
    X_test_with_country_df = pd.DataFrame(X_test_with_country, index=X_test_with_country.index, columns=X_test_with_country)
    # reassign pointers to purely quantitative features in X_train and X_test
    X_train = X_train_with_country.iloc[:,1:]
    X_test = X_test_with_country.iloc[:,1:]
    
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
    # scale training set and test set
    scaler = preprocessing.StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test) 
    scaled_X_train = pd.DataFrame(scaled_X_train, index=X_train.index, columns=X_train.columns)
    scaled_X_test = pd.DataFrame(scaled_X_test, index=X_test.index, columns=X_test.columns)
    
    # PART 1: FEATURE ENGINEERING 
    
    # INTERACTION TERM PAIRS
    print(DIVIDER + "INTERACTION TERM PAIRS" + DIVIDER)
    # degree of terms = 2
    # interaction_only means no feature multiplied by itself (x^2) only x*y
    # include_bias means including constant terms that act as intercept in linear model --> false
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    # numpy array of world.csv with 210 features
    world_int_term_pairs = poly.fit_transform(scaled_X_train) 
    int_pair_list = poly.get_feature_names(scaled_X_train.columns.tolist())
    int_term_pairs = pd.DataFrame({'features': int_pair_list})
    world_int_term_pairs_df = pd.DataFrame(data=world_int_term_pairs[:], columns=int_pair_list)
    print("Number of interaction term pairs:", len(int_term_pairs))
    print(int_term_pairs)
    print("\n")
   
    # CLUSTERING LABELS
    print(DIVIDER + "CLUSTERING LABELS" + DIVIDER)
    # use elbow method to determine suitable k value
    # within-cluster-sum-of-squares (WCSS)
    wcss = []
    # this loop will fit the k-means algorithm to data and compute the WCSS and append to list
    for i in range(1, 20):
        kmeans = KMeans(n_clusters=i, init='k-means++')
        kmeans.fit(scaled_X_train)
        # kmeans inertia attribute is sum of squared distance of samples to their closest cluster center
        wcss.append(kmeans.inertia_)
    
    # plot the elbow graph - choose n=5 for clustering as this is point where graph plateaus
    plt.figure(figsize=(12,6))
    plt.plot(range(1, 20), wcss, marker='o')
    plt.title('Elbow Method Graph for Selection of Appropriate k Value')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.savefig('task2bElbowGraph.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    
    # form clusters
    kmeans = KMeans(n_clusters=7) # any number between 5-7 should be good, but higher = not much difference
    clusters = kmeans.fit(scaled_X_train)
    prediction = kmeans.predict(scaled_X_train)
    
    # see count of data points in each cluster
    frame = pd.DataFrame(scaled_X_train)
    frame['k-means cluster'] = prediction
    counts_series = frame['k-means cluster'].value_counts().sort_index()
    counts_df = pd.DataFrame({'cluster number':counts_series.index, 'counts':counts_series.values})
    print("\nCount of countries in each cluster:")
    print(counts_df.to_string(index=False))
    # see which cluster each country from training set is in (should be a number between 0-6 because 7 clusters))
    print("\nCountries from training set with new feature (f-clusterlabels):")
    cluster_label_df = pd.DataFrame({'Country Code': X_train_with_country_df.iloc[:, 0], 'f-clusterlabel': clusters.labels_})
    print("\n", cluster_label_df.to_string(index=False))
    
    # Turn the 20 features from world.csv into 2 dimensions with PCA
    pca_2 = PCA(n_components=2)
    plot_columns = pca_2.fit_transform(scaled_X_train.iloc[:,:21])
    # Plot each cluster and shade by their cluster label 
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=scaled_X_train["k-means cluster"])
    plt.title('K-Means Clustering with 7 Clusters and 2 Principal Components')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('task2bKMeansClustering.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    # PART 2: FEATURE SELECTION
    
    # SELECT 4 FEATURES IN A PRINCIPLED MANNER
    print("\n" + DIVIDER + "SELECTING 4 FEATURES IN A PRINCIPLED MANNER" + DIVIDER)
    model = ExtraTreesClassifier(random_state=200)
    model.fit(scaled_X_train,y_train)
    #plot graph of top 10 most important features for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=scaled_X_train.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Most Important Features to Life Expectancy')
    plt.xlabel('Importance Score with Extra-Trees Classifier')
    plt.ylabel('Feature Name')
    plt.savefig('task2bTop10ImportantFeatures.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    print("The 4 chosen features and their relative importance (descending order) to predicting Life Expectancy are:")
    feat_importance_dict = feat_importances.nlargest(4).to_dict()
    # sort the dictionary in descending order of importance
    {key: value for key, value in sorted(feat_importance_dict.items(), key=lambda item: item[1], reverse=True)}
    dct_for_knn = {}
    dct_for_knn_test = {}
    i = 1
    for key, item in feat_importance_dict.items():
        dct_for_knn[key] = scaled_X_train[key]
        dct_for_knn_test[key] = scaled_X_test[key]
        print("Chosen feature number " + str(i) + ": " + key + " with an importance of " + str(round(item, 3)))
        i += 1
    principled_df = pd.DataFrame(dct_for_knn)
    principled_df_test = pd.DataFrame(dct_for_knn_test)
    
    
    # PCA
    print("\n" + DIVIDER + "PRINCIPAL COMPONENT ANALYSIS" + DIVIDER)
    pca = PCA(n_components=4)
    scaled_X_train = scaled_X_train.drop('k-means cluster', 1)
    pca.fit(scaled_X_train)
    principal_components = pca.transform(scaled_X_train)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC-1', 'PC-2', 'PC-3', 'PC-4'])
    principal_components_test = pca.transform(scaled_X_test)
    principal_test = pd.DataFrame(data=principal_components_test, columns=['PC-1', 'PC-2', 'PC-3', 'PC-4'])
    print("Reduced training set data from " + str(len(scaled_X_train.columns)) + " features to " + str(len(principal_df.columns)) + " Principal Components:")
    print(principal_df)
    
        
    # FIRST 4 FEATURES
    print("\n" + DIVIDER + "FIRST 4 FEATURES" + DIVIDER)
    first_4_features = scaled_X_train.iloc[:,:4]
    first_4_features_test = scaled_X_test.iloc[:,:4]
    print("The first 4 features of the original dataset are:")
    i = 1
    for column in first_4_features.columns:
        print(str(i) + "." + " " + column)
        i += 1

    
    # PART 3: PERFORM 5-NN CLASSIFICATION USING FEATURES SELECTED ABOVE
    print("\n==========ACCURACY OF EACH FEATURE GROUP IN 5-NN CLASSIFICATION==========")
    k5_classifier = KNeighborsClassifier(n_neighbors=5) 
    
    # 5-NN with 4 features selected in principled manner 
    k5_classifier.fit(principled_df, y_train)
    k5_test_accu_principled = k5_classifier.score(principled_df_test, y_test)
    
    # 5-NN with 4 features from PCA
    k5_classifier.fit(principal_df, y_train)
    k5_test_accu_PCA = k5_classifier.score(principal_test, y_test)
    
    # 5-NN with first 4 features
    k5_classifier.fit(first_4_features, y_train)
    k5_test_accu_first_4 = k5_classifier.score(first_4_features_test, y_test)
    print('Accuracy of feature engineering: {:.{width}f}'.format(k5_test_accu_principled, width=3))
    print('Accuracy of PCA: {:.{width}f}'.format(k5_test_accu_PCA, width=3))
    print('Accuracy of first four features: {:.{width}f}'.format(k5_test_accu_first_4, width=3))

task_2b()