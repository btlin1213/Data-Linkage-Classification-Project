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
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict
from nltk import FreqDist

THRESHOLD = 10

# stopwords removal prep
# first time:
# nltk.download('punkt')
# nltk.download('stopwords')
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
    

# Main Program
def task_1b():
    # initialise 2D array for csv 
    task1b_amazon_file = [['block_key', 'product_id']]
    task1b_google_file = [['block_key', 'product_id']]
    
    # read csv files as dataframe
    amazon_df = pd.read_csv("amazon.csv")
    google_df = pd.read_csv("google.csv")
    
    for index_a, row_a in amazon_df.iterrows():       
        processed_text = preprocessing(str(row_a['title'])) + preprocessing(str(row_a['description'])) + preprocessing(str(row_a['manufacturer']))                                                                                    
        fd = FreqDist(list(nltk.bigrams(processed_text)))
        for bigram_key, freq in fd.most_common(THRESHOLD):
            curr_row = [' '.join(bigram_key), row_a['idAmazon']]
            task1b_amazon_file.append(curr_row)
            
    for index_g, row_g in google_df.iterrows():       
        processed_text = preprocessing(str(row_g['name'])) + preprocessing(str(row_g['description'])) + preprocessing(str(row_g['manufacturer']))                                                                                    
        fd = FreqDist(list(nltk.bigrams(processed_text)))
        for bigram_key, freq in fd.most_common(THRESHOLD):
            curr_row = [' '.join(bigram_key), row_g['id']]
            task1b_google_file.append(curr_row)
                
    # writing 2D array to csv
    task1b_amazon_file = "\n".join([",".join(x) for x in task1b_amazon_file]) + "\n"
    task1b_google_file = "\n".join([",".join(y) for y in task1b_google_file]) + "\n"
    
    with open("amazon_blocks.csv", "w") as f1:
        f1.write(task1b_amazon_file)
    with open("google_blocks.csv", "w") as f2:
        f2.write(task1b_google_file)
        
        
task_1b()
