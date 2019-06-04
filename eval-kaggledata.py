#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import sys
from KaggleWord2VecUtility import KaggleWord2VecUtility

# Parameters
# ==================================================

tf.app.flags.DEFINE_string('f', '', 'kernel')

# Data Parameters
tf.flags.DEFINE_string("test_data_file", "data/labeledTrainData.tsv", "Data source for the test data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
<<<<<<< HEAD
tf.flags.DEFINE_string("checkpoint_dir", "runs/1559222525/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")
=======
tf.flags.DEFINE_string("checkpoint_dir", "runs/1559141057/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
>>>>>>> 1aedca2954e328bef692d8e5700c94088fdfc225

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels_kaggle(FLAGS.test_data_file)
    #y_test = np.argmax(y_test, axis=1
else:
    x_raw = []
    print("sentence? : ")
    keyword = input()
    x_raw.append(keyword)
    y_test = []

# preprocessing
sentences = []
# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
            
print("\nEvaluating...\n")    
# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0] 
        
        # Score derived from CNN
        scores = graph.get_operation_by_name("output/scores").outputs[0]

#x_test = np.array(list(vocab_processor.transform(x_raw)))        
#print("x_test : ",x_test.shape)

def evalu(each_sentence) :
    #print(each_sentence)
    each_sentence = np.array(list(vocab_processor.transform(each_sentence)))
    
    #print(each_sentence.shape)
    #print("before", each_sentence)
    
    batches = data_helpers.batch_iter(list(each_sentence),  1, 1,shuffle=False)
    
    for x_test_batch in batches:
        print("x_test_batch",x_test_batch)
        x_test_batch = np.array(x_test_batch)
        #print(x_test_batch.shape)
        
        return sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})[0], sess.run(scores, {input_x: x_test_batch, dropout_keep_prob: 1.0})[0]
        

                                  
# Collect the predictions here
all_predictions = []
softmax_score = np.array([0.0,0.0])
softmax_score = softmax_score.astype(np.float32)
count = 0

for x in x_raw : # 전체 말뭉치에서 검사
    print("count : ", count)
    count += 1
    x = KaggleWord2VecUtility.review_to_sentences(x) # 말뭉치에서 문장을 추출
    prediction=0 # 각 문장 별 긍정, 부정 레이블                    /
    score = [] # 각 문장 별 softmax 함수에 들어가기 전의 score
    scorelist=[] # 한 말뭉치에서의 score 리스트
    predictlist=[] # 한 말뭉치에서의 긍정, 부정 레이블
    
    for sentence in x: # 하나의 말뭉치에서 검사
        tmplist=[]
        tmplist.append(sentence)
        prediction ,score = evalu(tmplist) # 각 문장 별로 prediction과 score 도출
        scorelist.append(score) # score 리스트에 추가
       
<<<<<<< HEAD
        if score[0] < 0.0 and score[1] > 0.0:
            score[0] = 0.0 # softmax 함수 공식에 의해 음수는 영향이 아주 적으므로 무시 
        if score[0] > 0.0 and score[1] < 0.0 :
            score[1] = 0.0 # softmax 함수 공식에 의해 음수는 영향이 아주 적으므로 무시
        
=======
        if score[0] < 0.0 :
            score[0] = 0.0 # softmax 함수 공식에 의해 음수는 영향이 아주 적으므로 무시 
        if score[1] < 0.0 :
            score[1] = 0.0 # softmax 함수 공식에 의해 음수는 영향이 아주 적으므로 무시
            
>>>>>>> 1aedca2954e328bef692d8e5700c94088fdfc225
        np.add(softmax_score, score, out=softmax_score) # softmax score를 구함
        predictlist.append(prediction) # predict 리스트에 추가
        
    print("predict list: ", predictlist)
    print("score list: ", scorelist) 
    print("softmax_score: ", softmax_score)
    
    if softmax_score[0] > softmax_score[1] : # 부정 점수가 더 높은 경우
        all_predictions.append(0) # 부정 정답 마킹
    else : # 긍정 점수가 더 높은 경우
        all_predictions.append(1) # 긍정 정답 마킹
    
    softmax_score = np.array([0.0,0.0]) # 하나의 말뭉치를 전부 검사하면 점수 초기화
    
    
if FLAGS.eval_train is False:
    answer = all_predictions[0]
    print("answer : ", answer, "\n")
    if answer == 0 :
        print("\n negative!!")
    else :
        print("\n positive^^")
else:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

    # Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
