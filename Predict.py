import numpy as np
import tensorflow as tf
from tensorflow import keras

SA_RNN = keras.models.load_model("C:/Users/dashb/OneDrive/Documents/GitHub/Sentiment-Analysis/Model")


review = 'When i was 37 I went to the store and got my skittles taken. i am unhappy.'

prediction = SA_RNN.predict(np.array([review]))

if prediction >= 0.0:
    print("This review is positive", prediction)
else:
    print("This review is negetive", prediction)
