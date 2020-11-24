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
        '1/50': './assets/models/model_vul1_nonvul50.h5'
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
    print ("I loaded a model")
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
modelSelection = '1/50'
makePrediction('thisisTest huhuhuhuhuhuhuh brazil number one',modelSelection)
