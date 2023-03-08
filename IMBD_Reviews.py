import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
import glob

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

split = ['train[:70%]', 'train[70%:]']
trainDataset, testDataset = tfds.load(name='imdb_reviews', split=split, as_supervised=True)

BUFFER_SIZE = 10000
BATCH_SIZE = 128

train_dataset = trainDataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = testDataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)

print(train_dataset)

encoder.adapt(train_dataset.map(lambda text, label: text))

print(train_dataset)


def RNN():
    SA_RNN = tf.keras.models.Sequential([
        encoder,
        tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim = 128, mask_zero = True),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    return SA_RNN

SA_RNN = RNN()

SA_RNN.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])

SA_RNN.fit(train_dataset, epochs = 20)

SA_RNN.save('C:/Users/dashb/OneDrive/Documents/GitHub/Sentiment-Analysis/Model')

