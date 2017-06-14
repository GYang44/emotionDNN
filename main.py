from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

# Data sets
Emotion_TRAINING = "train.csv"

Emotion_TEST = "test.csv"

def main():
  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=Emotion_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=Emotion_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Specify that all features have real-value data
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=136)]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[272, 544, 272],
                                              n_classes=8,
                                              model_dir="/tmp/emotion_model")
  
  # Define the training inputs
  def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)

    return x, y
  

  # Fit model.
  classifier.fit(input_fn=get_train_inputs, steps=12000)

  # Saving model
  #model_weight = tf.Variable([1,2,3],[272,544,272], dtype=tf.float32, name='weights')
  #model_bias = tf.Variable([1,2,3], dtype=tf.float, name='bias')

  # Define the test inputs
  def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)

    return x, y

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                       steps=1)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Classify two new flower samples.
#  def new_samples():
#    return np.array(
#      [[6.4, 3.2, 4.5, 1.5],
#       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

#  predictions = list(classifier.predict(input_fn=new_samples))

#  print(
#      "New Samples, Class Predictions:    {}\n"
#      .format(predictions))

if __name__ == "__main__":
    main()
