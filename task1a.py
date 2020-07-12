import os
import csv
import nltk
import random
import datetime
import collections
import textdistance
import pandas as pd
import numpy as np
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from string import digits

# constant definitions
STRING_METRICS_COUNT = 3
THRESHOLD = 0.3

# stopwords removal prep
# first time:
#nltk.download('punkt')
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# stemming prep 
ps = PorterStemmer()
# tokenizing prep 
tokenizer = RegexpTokenizer(r'\w+')


# Preprocessing Function

def preprocessing(string):
    # remove numbers
    no_numbers = str.maketrans('', '', digits)
    string = string.translate(no_numbers)
    # remove single letters
    string = ' '.join([char for char in string.split() if len(char)>1])
    # case folding
    string.casefold()
    # tokenise
    word_list = tokenizer.tokenize(string)
    # remove stop words and applying stemming
    filtered_list = [ps.stem(word) for word in word_list if not word in stop_words]    
    # return cleaned string as a list of stem words
    return filtered_list
    

# Similarity Evaluation Functions

def similarity_score(word_list1, word_list2):
    string1 = " ".join(word_list1)
    string2 = " ".join(word_list2)
    sorensen = textdistance.sorensen(word_list1, word_list2)
    cosine = textdistance.cosine(word_list1, word_list2)
    ratcliff = textdistance.ratcliff_obershelp.normalized_similarity(string1, string2)
    return (sorensen + cosine + ratcliff) / STRING_METRICS_COUNT
    

def google_first_iteration(df):
    google_id = df.idGoogleBase.tolist()
    google_titles = df.name.tolist()
    google_titles = [preprocessing(google_title) for google_title in google_titles]  
    return zip(google_id, google_titles)


# Main Program

def task_1a():
    # initialise 2D array for csv 
    task1a_file = [['idAmazon', 'idGoogleBase']]

    processed_google = {}
    
    # read csv files as dataframe
    amazon_df = pd.read_csv("amazon_small.csv")
    google_df = pd.read_csv("google_small.csv")
    
    amazon_id = amazon_df.idAmazon.tolist()
    google_id = google_df.idGoogleBase.tolist()

    amazon_titles = amazon_df.title.tolist()
    amazon_titles = [preprocessing(a_title) for a_title in amazon_titles]
    
    google_titles = google_df.name.tolist()
    google_titles = [preprocessing(g_title) for g_title in google_titles]

    
    for a_id, a_title in zip(amazon_id, amazon_titles):
        max_score = 0
        for g_id, g_title in zip(google_id, google_titles):
            score = similarity_score(a_title, g_title)
            if score > max_score:
                max_score = score
                max_link = g_id
        if max_score > THRESHOLD:
            curr_row = [a_id, max_link]
            task1a_file.append(curr_row)
        
    
#     writing 2D array to csv
    task1a_file = "\n".join([ ",".join(x) for x in task1a_file ]) + "\n"
    with open("task1a.csv", "w") as f:
        f.write(task1a_file)

        
task_1a()


#  for index_a, row_a in amazon_df.iterrows():
#         # reset max score
#         max_score = 0
#         max_link = ""
#         # compile content
#         content_a = str(row_a['title']) + str(row_a['description']) + str(row_a['manufacturer'])
#         # preprocess content
#         word_list1 = preprocessing(content_a)
