import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
from tensorflow import keras
from keras.models import load_model

from datetime import datetime
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers

def switch (modelSelection):
    switcher = {
        '1/50': './assets/models/conv_vul1_nonvul50.h5',
        '1/30': './assets/models/conv_vul1_nonvul30.h5',
        '1/1': './assets/models/conv_vul1_nonvul1.h5',
        '50/1': './assets/models/conv_vul50_nonvul1.h5',
        '30/1': './assets/models/conv_vul30_nonvul1.h5',
        '1/100': './assets/models/conv_vul1_nonvul100.h5',
        '100/1': './assets/models/conv_vul100_nonvul1.h5',
        'lstm30/1':'./assets/models/lstm_vul30_nonvul1.h5'
        'lstm1/1':'./assets/models/lstm_vul1_nonvul1.h5'
        'original' : './assets/models/original.h5'
    }
    return switcher.get(modelSelection,'invalidModel')

def makePrediction(messages_as_string, modelSelection):
    print ("Running prediction function...")
    messages = list(messages_as_string.split('s3cur!tywh@l3'))

    vocab_size = 15613 # amount of words that appear in commit messages
    sequence_length = 1000 # arbitrary vector length

    embedding_layer = tf.keras.layers.Embedding(vocab_size, sequence_length)

    # Use the text vectorization layer to normalize, split, and map strings to
    # integers. Note that the layer uses the custom standardization defined above.
    # Set maximum_sequence length as all samples are not of the same length.
    vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=sequence_length)
    text_ds = tf.data.Dataset.from_tensor_slices(messages).batch(32)
    vectorizer.adapt(text_ds)

    #switch case for select
    path = switch(modelSelection)
    if path == 'invalidModel': return path
    #path = './assets/models/' + modelSelection
    print ("trying to load model at: " + path)
    model = load_model(path)
    string_input = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(string_input)
    preds = model(x)
    end_to_end_model = keras.Model(string_input, preds)

    count  = 0
    vulnProbabilitySum = 0
    nonVulnProbabilitySum = 0

    for message in messages:
        count = count + 1
        probabilities = end_to_end_model.predict(
            [[message]]
            )
        print(message)
        vulnProbabilitySum = vulnProbabilitySum + probabilities[0][1]
        print('vuln:',probabilities[0][1])
        nonVulnProbabilitySum = nonVulnProbabilitySum + probabilities[0][0]
        print('nonvuln:',probabilities[0][0])
    vulnLikelyHoodStr = '0' if count == 0 else str(vulnProbabilitySum/count)
    nonVulnLikelyHoodStr = '0' if count == 0 else str(nonVulnProbabilitySum/count)
    ## confidence should equal approximately 1
    confindence = '0' if count == 0 else str(vulnProbabilitySum/count + nonVulnProbabilitySum/count)
    return_string = vulnLikelyHoodStr + ',' + nonVulnLikelyHoodStr + ',' + confindence
    print("Response body: \n" + return_string)

    return return_string

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation), '')

def testModels():
    data = pd.read_csv("assets/csv-data/sample-data.csv")

    # test train split
    msk = np.random.rand(len(data)) < 0.8
    train_X = data[msk]['message']
    train_Y = data[msk]['vulnerable']
    test_X  = data[~msk]['message']
    test_Y  = data[~msk]['vulnerable']

    max = 0
    mess = ""

    for message in train_X:

        if len(message) > max:
            max = len(message)
            mess = message

    vocab_size = 15613
    sequence_length = 1000
    embedding_layer = tf.keras.layers.Embedding(vocab_size, sequence_length)
    vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=sequence_length)
    text_ds = tf.data.Dataset.from_tensor_slices(train_X).batch(32)
    vectorizer.adapt(text_ds)


    # Use the text vectorization layer to normalize, split, and map strings to
    # integers. Note that the layer uses the custom standardization defined above.
    # Set maximum_sequence length as all samples are not of the same length.
    models = {'1/1','1/50','50/1','1/30','30/1','1/100','100/1','original'}
    modelVulnConsensus = 0
    modelNonVulnConsensus = 0

    for i in models:
        #switch case for select
        path = switch(i)
        if path == 'invalidModel': return path
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print ("trying to load model at: " + path)
        print (i + " model loaded successfully.")
        print("-------------------------------------------")
        if i == 'original' :
            vocab_size = 12612
            embedding_layer = tf.keras.layers.Embedding(vocab_size, sequence_length)
            vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=sequence_length)
            text_ds = tf.data.Dataset.from_tensor_slices(train_X).batch(32)
            vectorizer.adapt(text_ds)
            print("The Original Model was trained using a vocabulary size of 12612 instead of 15613")
        else :
            # Use the text vectorization layer to normalize, split, and map strings to
            # integers. Note that the layer uses the custom standardization defined above.
            # Set maximum_sequence length as all samples are not of the same length.
            vocab_size = 15613
            embedding_layer = tf.keras.layers.Embedding(vocab_size, sequence_length)
            vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=sequence_length)
            text_ds = tf.data.Dataset.from_tensor_slices(train_X).batch(32)
            vectorizer.adapt(text_ds)
        model = load_model(path)
        string_input = keras.Input(shape=(1,), dtype="string")
        x = vectorizer(string_input)
        preds = model(x)
        end_to_end_model = keras.Model(string_input, preds)
        count = 0
        Vuln = 0
        vulnLengthSum = 0
        nonVuln = 0
        nonVulnLengthSum = 0
        vulnProbabilitySum = 0
        nonVulnProbabilitySum = 0
        #print("non-vulnerables:")
        for x in test_X:
            count = count + 1
            probabilities = end_to_end_model.predict(
                [[x]]
                )
            np.argmax(probabilities[0])
            vulnProbabilitySum = vulnProbabilitySum + probabilities[0][1]
            nonVulnProbabilitySum = nonVulnProbabilitySum + probabilities[0][0]
            if probabilities[0][1] > 0.5 :
                vulnLengthSum = vulnLengthSum + len(x)
                #print("length:",len(x))
                #print(x)
                Vuln = Vuln + 1
            if probabilities[0][0] > 0.5 :
                nonVulnLengthSum = nonVulnLengthSum + len(x)
                #print("length:",len(x))
                #print(x)
                nonVuln = nonVuln + 1
        print("Length test:")
        print("Vuln commits (over 50% likely to be vulnerable): ", Vuln)
        print("Average vuln length",  vulnLengthSum/Vuln)
        print("non vuln commits(under 50% likely to be vulnerable): ", nonVuln)
        print("Average non vuln length", nonVulnLengthSum/nonVuln)
        vulnLikelyHoodStr = '0' if count == 0 else str(vulnProbabilitySum/count)
        nonVulnLikelyHoodStr = '0' if count == 0 else str(nonVulnProbabilitySum/count)
        ## confidence should equal approximately 1
        confindence = '0' if count == 0 else str(vulnProbabilitySum/count + nonVulnProbabilitySum/count)
        return_string = vulnLikelyHoodStr + ',' + nonVulnLikelyHoodStr + ',' + confindence
        print("---------------------------------------------------")
        print("Response body sent to web app: \n" + return_string)
        modelVulnConsensus += vulnProbabilitySum/count
        modelNonVulnConsensus += nonVulnProbabilitySum/count
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    modelVulnConsensus = modelVulnConsensus/len(models)
    modelNonVulnConsensus = modelNonVulnConsensus/len(models)
    print("On average, the models predict that this set of data is:")
    print(str(modelVulnConsensus) + " Percent Vulnerable")
    print(str(modelNonVulnConsensus) + " Percent Non-Vulnerable")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
testModels()
