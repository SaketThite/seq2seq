# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 17:00:49 2018

@author: SA010TH
"""
import numpy as np
import tensorflow as tf
import re
import time
import pandas as pd
import os
import gensim
import re
from gensim.models import word2vec


'''
clean_questions = []
clean_answers = []
questions_into_int = []
answers_into_int = []
sorted_clean_questions = []
sorted_clean_answers = []
word2count = {}
questionswords2int = {}
answerswords2int = {}
questionswords2int = {}
answerswords2int = {}
answersints2word = {}
questionsints2word = {}
q_embedding_matrix = {} 
a_embedding_matrix = {} 
qi_embedding_matrix = {}
ai_embedding_matrix = {}

epochs = 100
batch_size = 64
rnn_size = 256
num_layers = 3
encoding_embedding_size = 50
decoding_embedding_size = 50
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5
output_max_length = 40
decoder_embedded_input = []

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

'''

# Doing a first cleaning of the texts


def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"werent", "were not", text)

    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

# Creating placeholders for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob

# Preprocessing the targets
def preprocess_targets(targets, word2int):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets

# Creating the Encoder RNN

def load_vocab(model, sentences):
    
    #print(type(self.input_texts))
    
    #sentences = self.input_texts
    

    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    #model = word2vec.Word2Vec(sentences.reshape(-1, 1), iter=5, min_count=1, size=5, workers=4)  'GoogleNews-vectors-negative300.bin'

    
    # get the most common words
    #print("most common words  :: \n ", model.wv.index2word[0:10])
        
    # get the least common words
    #print("least common words  :: \n ", model.wv.index2word[vocab_size - 1:vocab_size-10])
       
# convert the wv word vectors into a numpy matrix 
    
    embedding_matrix = {} 
    sentences = sentences.reshape(-1, 1)
    for line in sentences:
        words = line.split(" ")
        for word in words:
            embedding_vector = model[word]
            if embedding_vector is not None:
               embedding_matrix[word] = embedding_vector
                
        
    
    #for i, word in enumerate(model.wv.vocab):
        
    #    embedding_vector = model.wv[model.wv.index2word[i]]
    #    if embedding_vector is not None:
    #        embedding_matrix[word] = embedding_vector
     
    
    #print("embedding matrix :: \n ", embedding_matrix['room'])
    return embedding_matrix

def loadModel():
    model = gensim.models.KeyedVectors.load_word2vec_format("glove.6B.50dw2vformat.txt", binary=False)
    vocab_size = len(model.wv.vocab)
    print("Vocab size :: ",vocab_size)
    
    return model

def loadAndCleanQAs():
    lines = pd.read_csv('SampleData.csv', encoding="utf-8")
    #conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
    
    questions = lines["SentimentText"]
    answers = lines["ResponseText"]

    # Cleaning the questions
    for question in questions:
        clean_questions.append(clean_text(question))
    # Cleaning the answers
    for answer in answers:
        clean_answers.append(clean_text(answer))

    # Adding the End Of String token to the end of every answer
    for i in range(len(clean_answers)):
        clean_answers[i] += ' <EOS>'


    return    

def stripSpecialChars(word):
    
    word = re.sub('[^A-Za-z0-9]+', '', word)
    
    return word

def countAndFilterWords(model):

    # Creating a dictionary that maps each word to its number of occurrences
    for question in clean_questions:
        for word in question.split():
            word = stripSpecialChars(word)
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1
                
    for answer in clean_answers:
        for word in answer.split():
            word = stripSpecialChars(word)
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1
        
    # Creating two dictionaries that map the questions words and the answers words to a unique integer
    threshold_questions = 3

    word_number = 0
    
    for word, count in word2count.items():
        if count >= threshold_questions:
            questionswords2int[word] = word_number
            try:    
                embedding_vector = model[word]
                q_embedding_matrix[word] = embedding_vector
                qi_embedding_matrix[word_number] = embedding_vector
            except:
                print("Ignore word" + word)
            word_number += 1
    threshold_answers = 3
    
    word_number = 0
    
    for word, count in word2count.items():
        if count >= threshold_answers:
            answerswords2int[word] = word_number
            try:    
                embedding_vector = model[word]
                a_embedding_matrix[word] = embedding_vector
                ai_embedding_matrix[word_number] = embedding_vector
            except:
                print("Ignore word" + word)
            word_number += 1
            
        # Adding the last tokens to these two dictionaries
    tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
    for token in tokens:
        questionswords2int[token] = len(questionswords2int) + 1
    for token in tokens:
        answerswords2int[token] = len(answerswords2int) + 1

            
    return

def sortQnAs():
    # Translating all the questions and the answers into integers
    # and Replacing all the words that were filtered out by <OUT> 
    for question in clean_questions:
        ints = []
        for word in question.split():
            if word not in questionswords2int:
                ints.append(questionswords2int['<OUT>'])
            else:
                ints.append(questionswords2int[word])
        questions_into_int.append(ints)
    
    for answer in clean_answers:
        ints = []
        for word in answer.split():
            if word not in answerswords2int:
                ints.append(answerswords2int['<OUT>'])
            else:
                ints.append(answerswords2int[word])
        answers_into_int.append(ints)
     
    # Sorting questions and answers by the length of questions

    for length in range(1, 25 + 1):
        for i in enumerate(questions_into_int):
            if len(i[1]) == length:
                sorted_clean_questions.append(questions_into_int[i[0]])
                sorted_clean_answers.append(answers_into_int[i[0]])
                
    return

            

def createInverseDictionaries():

    #answersints2word = {w_i: w for w, w_i in answerswords2int.items()}
    
    for k, v in answerswords2int.items():
        answersints2word[v] = k

    for k, v in questionswords2int.items():
        questionsints2word[v] = k

    #questionsints2word = {w_i: w for w, w_i in questionswords2int.items()}
    
    return

def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    #lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    #encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.dynamic_rnn(lstm, rnn_inputs, dtype=tf.float32)
    
    #tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
    #                                                                cell_bw = encoder_cell,
    #                                                                sequence_length = sequence_length,
    #                                                                inputs = rnn_inputs,
    #                                                                dtype = tf.float32)
    return encoder_output, encoder_state

def decode(helper, encoder_outputs, sequence_length, num_words):
    
    print("5")
    try:    
        num_units = 50
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    
        #decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
        
        #attention_mech = tf.contrib.seq2seq.BahdanauAttention(num_units=num_units, 
        #                                                  memory=attention_states,
        #                                                  memory_sequence_length=sequence_length)
        
        #attention_mech = tf.contrib.seq2seq.LuongAttention(
        #                        num_units, attention_states,
        #                            memory_sequence_length=sequence_length)
        
        print("6")
    
        #decoder_cell = tf.contrib.seq2seq.AttentionWrapper( 
        #        lstm_cell, attention_mech, attention_layer_size=num_units / 2)
        
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
    
        out_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, num_words, reuse=False) 
    
        print("7")
    
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=out_cell, helper=helper,
                                                  initial_state=out_cell.zero_state(dtype=tf.float32, 
                                                                                    batch_size=batch_size))
        
                
    
        outputs = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False,
                                                    impute_finished=True, maximum_iterations=output_max_length)
        
        return outputs[0]

    
    except Exception as e:
        print(str(e))

    print("8")
        


# Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, encoder_output, num_words, sequence_length, word2int):
            
    print("4")
    try:
        
        decoder_inputs_length = tf.placeholder(
                    dtype=tf.int32, shape=(None,), name='decoder_inputs_length')
        
        decoder_inputs_length_train = decoder_inputs_length + 1
    
    
        train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_embedded_input, decoder_inputs_length_train)
    
        training_predictions = decode(train_helper, encoder_output, sequence_length, num_words)
    
        start_tokens = tf.zeros([batch_size], dtype=tf.int32)

        rank1Tensor = tf.Variable(list(qi_embedding_matrix.keys()), tf.int32)

        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(qi_embedding_matrix, start_tokens=start_tokens, 
                                                               end_token=word2int['<EOS>'])
        
        test_predictions = decode(pred_helper, encoder_output, sequence_length, num_words)
            
        return training_predictions, test_predictions

    except Exception as e:
        print(str(e))


#Build the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    
    encoder_outputs, encoder_state = encoder_rnn(encoder_embedded_input, 
                                                 rnn_size, 
                                                 num_layers, 
                                                 keep_prob, 
                                                 sequence_length)
    
   
    preprocessed_targets = preprocess_targets(targets, questionswords2int)
    
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         #decoder_embeddings_matrix,
                                                         encoder_outputs,
                                                         questions_num_words,
                                                         sequence_length,
                                                         questionswords2int)
    return training_predictions, test_predictions

def buildData():
    model = loadModel()
    
    loadAndCleanQAs()
    
    # Importing the dataset
    #a_embed_matrix = load_vocab(model, clean_answers)
    countAndFilterWords(model)
    
    createInverseDictionaries()
    
    sortQnAs()
    return

def trainAndTestModel():
    
    tf.reset_default_graph()
    session = tf.InteractiveSession()
     
    # Loading the model inputs
    inputs, targets, lr, keep_prob = model_inputs()
    
    # Getting the shape of the inputs tensor
    input_shape = tf.shape(inputs)
     
    # Setting the sequence length
    sequence_length = tf.placeholder_with_default(50, None, name = 'sequence_length')
     
     
    # Getting the training and test predictions
    
# Getting the training and test predictions
    training_predictions, test_predictions = seq2seq_model(inputs,
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int)
    
    return

def main():
    
    #buildData()
    
    trainAndTestModel() 

    return

main()

test = list(qi_embedding_matrix.keys())

print(type(test))


    
    
