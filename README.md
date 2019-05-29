# CNN을 이용한 자연어 처리에 기반한 영화 평론 데이터에서의 감성 인식 
**영어 말뭉치**에서 nltk 라이브러리를 통해 단어 단위로 전처리 된 '로튼 토마토 영화 리뷰' 데이터를 **CNN 모델**을 통해 긍정, 부정을 분류하는 프로젝트

## 1. Model Structure
![Model](http://www.wildml.com/wp-content/uploads/2015/11/Screen-Shot-2015-11-06-at-8.03.47-AM.png)
      출처 : http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

1. 정답 레이블이 있는 '로튼 토마토 영화 리뷰' 데이터 75249건((https://github.com/e9t/nsmc))에 대해서 **품사 태깅**

2. 품사 태깅한 단어들에 대해 **Word2Vec**을 이용해 학습시킨 임베딩 벡터로 변환

3. 단어 벡터들을 **BiLSTM**에 넣어서 양쪽 끝 state들에 대해서 **fully connected layer**와 **Softmax**함수를 이용해 분류

## 2. Requirement

- [nltk](https://datascienceschool.net/view-notebook/118731eec74b4ad3bdd2f89bab077e1b/)
- [tensorflow 1.13.1](https://www.tensorflow.org/)

## 3. Data Sets

- Training data : 영화 리뷰 데이터 (-) 36420 [rt-polarity.neg](https://github.com/HyungjinLee/NLP_Text_Classification/tree/master/rt-polaritydata)
                  영화 리뷰 데이터 (+) 38829 [rt-polarity.pos]
(https://github.com/HyungjinLee/NLP_Text_Classification/tree/master/rt-polaritydata)
                  총 리뷰 데이터 = 75249 문장

- Test data : Kaggle 리뷰 데이터 말뭉치 5만건 [ratings_test.txt](https://github.com/e9t/nsmc)

## 4. 학습

![alt text](https://github.com/MSWon/Sentimental-Analysis/blob/master/pic/pic_2.png "Word2Vec Tensorboard")

1. Sentimental-Analysis 폴더를 github로부터 다운로드

2. **Word2Vec_train.py**로 품사 태깅한 단어들에 대해서 Word2Vec 학습 후 모델 저장 [Word2vec.model](https://drive.google.com/file/d/1Jxf_F_ibneTNRe_4glcWTYmj0TgLh8fP/view?usp=sharing)

3. **Word2Vec_Tensorboard.py**를 통해 시각화

4. cmd창에 cd ./Sentimental-Analysis-master/Sentimental-Analysis-master/Bidirectional_LSTM 경로로 이동

5. **python Bi_LSTM_train.py** 명령어를 통해 이진 분류기 학습

![alt text](https://github.com/MSWon/Sentimental-Analysis/blob/master/pic/pic_4.png "Accuracy graph")

   **epoch 4 이후에 overfitting이 되므로 epoch 4에서 early stopping을 한다.**

## 5. 결과

![alt text](https://github.com/MSWon/Sentimental-Analysis/blob/master/pic/pic_3.png "Result table")

- Bi_LSTM_test.py를 통해 test data에 대해서 성능 확인 (**86.52%**)

- Doc2Vec, Term-existance Naive Bayes에 의한 성능 보다 뛰어남([박은정](https://www.slideshare.net/lucypark/nltk-gensim))


![alt text](https://github.com/MSWon/Sentimental-Analysis/blob/master/pic/pic_5.png "Result")


- Grade_review.py를 통해 직접 입력한 문장에 성능 확인
