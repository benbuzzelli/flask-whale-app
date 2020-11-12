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

def makePrediction(messages_as_string):
    print ("Running prediction function...")
    messages = list(messages_as_string.split('s3cur!tywh@l3'))

    vocab_size = 12612
    sequence_length = 1000

    embedding_layer = tf.keras.layers.Embedding(vocab_size, sequence_length)

    # Use the text vectorization layer to normalize, split, and map strings to
    # integers. Note that the layer uses the custom standardization defined above.
    # Set maximum_sequence length as all samples are not of the same length.
    vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=sequence_length)
    text_ds = tf.data.Dataset.from_tensor_slices(messages).batch(32)
    vectorizer.adapt(text_ds)

    path = './assets/models/model.h5'
    print ("trying to load model at: " + path)
    model = load_model(path)
    print ("I loaded a model")

    string_input = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(string_input)
    preds = model(x)
    end_to_end_model = keras.Model(string_input, preds)

    count  = 0
    Vuln = 0
    vulnLengthSum = 0
    nonVuln = 0
    nonVulnLengthSum = 0

    for message in messages:
        count = count + 1
        probabilities = end_to_end_model.predict(
            [[message]]
            )
        np.argmax(probabilities[0])

        if probabilities[0][1] > 0.5 :
            vulnLengthSum = vulnLengthSum + len(message)
            # print("length:",len(message))
            # print(message)
            Vuln = Vuln + 1
        if probabilities[0][0] > 0.5 :
            nonVulnLengthSum = nonVulnLengthSum + len(message)
            # print("length:",len(message))
            # print(message)
            nonVuln = nonVuln + 1

    vuln = str(Vuln)
    avg_vuln = '0' if vulnLengthSum == 0 else str(vulnLengthSum/Vuln)
    isVuln = 'true' if Vuln > nonVuln else 'false'
    non_vuln = str(nonVuln)
    avg_non_vuln = '0' if nonVulnLengthSum == 0 else str(nonVulnLengthSum/nonVuln)
    return_string = vuln + "," + non_vuln + "," + isVuln

    print("Response body: \n" + return_string)

    return return_string
