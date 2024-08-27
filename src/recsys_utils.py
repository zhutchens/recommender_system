import numpy as np
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from string import punctuation
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, root_mean_squared_error, r2_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt 
from sentence_transformers import SentenceTransformer
from rake_nltk import Rake


def build_profile(items: list) -> np.array:
    # shapes = [array.shape[1] for array in items]
    # max_shape = max(shapes)

    # user_vector = np.zeros((1, max_shape))
    user_vector = np.zeros((1, items[0].shape[1]))
    for lst in items:
        # lst = np.pad(lst, ((0, 0), (0, max_shape - lst.shape[1])), 'constant')
        user_vector = np.add(user_vector, lst) # sum item vectors into one 
    
    return np.divide(user_vector, len(items)) # compute average of all items user interacted with 

        
def anaylze_item(item_description: str, num_keywords: int, bs: int) -> np.ndarray:
    '''
    Process and vectorize item descriptions using sentence transformer

    Arguments:
        item_description (str): description of item to vectorize
        num_keywords (int): number of keywords to extract from description
        bs (int): batch size to be used with the sentence transformer  
    ''' 

    rake = Rake()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    stop_words = stopwords.words('english')

    item_description = ''.join([char for char in item_description if char not in punctuation])

    item_description = word_tokenize(item_description)
    item_description = [word.lower() for word in item_description if word.lower() not in stop_words]
    item_description = ' '.join([word for word in item_description])

    rake.extract_keywords_from_text(item_description)
    sentences = rake.get_ranked_phrases()
    
    keywords = []
    for sentence in sentences: # may be more than one sentence, so loop just in case 
        for word in sentence.split():
            keywords.append(word)

    keywords = ' '.join([word for word in keywords][:num_keywords])

    return model.encode(keywords, batch_size = bs)


def train_user_model(user_features, review_scores, learning_task, verbose = False):
    x_train, x_test, y_train, y_test = train_test_split(user_features, review_scores, test_size = 0.1)
    
    if learning_task.lower() == 'regression':
        model = DecisionTreeRegressor()
    else:
        model = DecisionTreeClassifier()

    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    if verbose:
        print("=========== EVALUATION ===========")
        if learning_task.lower() == 'regression':
            print(f'RMSE: {root_mean_squared_error(y_test, preds)}')
            print(f'R2_SCORE: {r2_score(y_test, preds)}')
        else:
            print(f'ACCURACY: {accuracy_score(y_test, preds)}')
            print(f'PRECISION: {precision_score(y_test, preds)}')
            print(f'RECALL: {recall_score(y_test, preds)}')
            
            ConfusionMatrixDisplay(confusion_matrix(y_test, preds)).plot()
            plt.show()

    return preds, y_test


def predict_score(item):
    pass
    

# def filtering(user, items, n):
    # returns top n suggested items for user



    





