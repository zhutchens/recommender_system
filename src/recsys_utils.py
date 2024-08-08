import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from rake_nltk import Rake
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer

# def build_profile(user_id, items):

        
def anaylze_item(item_description: str) -> np.ndarray:
    '''
    Process, extract keywords, and vectorize item descriptions using the BERT Sentence Transformer or TF-IDF Vectorization. 
    Default TF-IDF. 
    ''' 

    stop_words = stopwords.words('english')
    rake = Rake()

    item_description = ''.join([char for char in item_description if char not in punctuation])

    item_description = word_tokenize(item_description)
    item_description = [word.lower() for word in item_description if word.lower() not in stop_words]
    item_description = ' '.join([word for word in item_description])

    return TfidfVectorizer().fit_transform([item_description]).toarray()


# def filtering(user, items, n):
    # returns top n suggested items for user 



    






