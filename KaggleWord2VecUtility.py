import numpy as np
import re
import pandas as pd
from KaggleWord2VecUtility import KaggleWord2VecUtility

<<<<<<< HEAD
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from multiprocessing import Pool
=======
>>>>>>> 1aedca2954e328bef692d8e5700c94088fdfc225

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

<<<<<<< HEAD
    @staticmethod
    def review_to_wordlist(review, remove_stopwords=False):
        # 1. HTML 제거
        review_text = BeautifulSoup(review, "lxml").get_text()   
        # 2. 특수문자를 공백으로 바꿔줌
        review_text = re.sub('[^a-zA-Z]', ' ', review_text)
        # 3. 소문자로 변환 후 나눈다.
        words = review_text.split()
        # 4. 불용어 제거
        if remove_stopwords:
            stops = set(stopwords.words('english'))
            words = [w for w in words if not w in stops]
        # 5. 어간추출
        #stemmer = SnowballStemmer('english')
        #words = [stemmer.stem(w) for w in words]
        
        # 6. 리스트 형태로 반환
        return(words)
=======
>>>>>>> 1aedca2954e328bef692d8e5700c94088fdfc225

def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

<<<<<<< HEAD
    @staticmethod
    def review_to_sentences( review, remove_stopwords=False ):
        # punkt tokenizer를 로드한다.
        """
        이 때, pickle을 사용하는데
        pickle을 통해 값을 저장하면 원래 변수에 연결 된 참조값 역시 저장된다.
        저장된 pickle을 다시 읽으면 변수에 연결되었던
        모든 레퍼런스가 계속 참조 상태를 유지한다.
        """
        
        # 1. nltk tokenizer를 사용해서 문장으로 토큰화 하고 공백 등을 제거한다.
        raw_sentences = sent_tokenize(review.strip())
        #print("raw_sentences =", raw_sentences)
        # 2. 각 문장을 순회한다.
        sentences = []
        for raw_sentence in raw_sentences:
            # 비어있다면 skip
            if len(raw_sentence) > 0:
                # 태그제거, 알파벳문자가 아닌 것은 공백으로 치환, 불용어제거
                #sentences.append(\
                #    KaggleWord2VecUtility.review_to_wordlist(\
                #    raw_sentence, remove_stopwords))
                raw_sentence = re.sub('[^\\-a-zA-Z]', ' ', raw_sentence)
        return raw_sentences
    
    @staticmethod
    def review_to_corpus( review, remove_stopwords=False ):
        # punkt tokenizer를 로드한다.
        """
        이 때, pickle을 사용하는데
        pickle을 통해 값을 저장하면 원래 변수에 연결 된 참조값 역시 저장된다.
        저장된 pickle을 다시 읽으면 변수에 연결되었던
        모든 레퍼런스가 계속 참조 상태를 유지한다.
        """
        step1 = BeautifulSoup(review, "lxml") #html 태그 제거
        step2 = re.sub('[^a-zA-z.]', ' ', step1.get_text()) # 소문자와 대문자가 아닌 것은 공백으로 대체
        step3 = step2.lower() #모두 소문자로 변환
        return step3

    # 참고 : https://gist.github.com/yong27/7869662
    # http://www.racketracer.com/2016/07/06/pandas-in-parallel/
    # 속도 개선을 위해 멀티 스레드로 작업하도록
    @staticmethod
    def _apply_df(args):
        df, func, kwargs = args
        return df.apply(func, **kwargs)
=======
def load_data_and_labels_kaggle(test_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    y=[]
    # Load data from files
    test_examples = pd.read_csv(test_data_file, 
                    header=0, delimiter='\t', quoting=3)
    # Generate labels
    for x in test_examples["sentiment"]:
        if x == 1 : #positive
            test_labels = [1]
        else : #negative
            test_labels = [0]
        y = np.concatenate([y,test_labels], 0)
    print(y)
    print(y.shape)
    print("sentiment complete")
    #print(test_examples["review"][:10])
    # preprocessing
    sentences = []
    for review in test_examples["review"]:
        tmpstr = KaggleWord2VecUtility.review_to_corpus(review, remove_stopwords=False)
        sentences.append(tmpstr)
    
    print("preprocessing complete")
    return [sentences, y]
>>>>>>> 1aedca2954e328bef692d8e5700c94088fdfc225

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
