from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

def main():
  with tf.Session() as sess:
    saver = tf.train.import_meta_graph('../emotion_model/model.ckpt-15000.meta')
    saver.restore(sess, '../emotion_model/model.ckpt-15000')
    #print(sess.run('bias:0'))
    graph = tf.get_default_graph()

    ##print all variables in the model
    #print(tf.all_variables())

    #weights for layer 0, 1, 2 and logits
    w0 = graph.get_tensor_by_name("dnn/hiddenlayer_0/weights/part_0:0")
    w0.s = graph.get_tensor_by_name("dnn/hiddenlayer_0/weights/part_0:0").shape
    w1 = graph.get_tensor_by_name("dnn/hiddenlayer_1/weights/part_0:0")
    w1.s = graph.get_tensor_by_name("dnn/hiddenlayer_1/weights/part_0:0").shape
    w2 = graph.get_tensor_by_name("dnn/hiddenlayer_2/weights/part_0:0")
    w2.s = graph.get_tensor_by_name("dnn/hiddenlayer_2/weights/part_0:0").shape
    wL = graph.get_tensor_by_name("dnn/logits/weights/part_0:0")
    wL.s = graph.get_tensor_by_name("dnn/logits/weights/part_0:0").shape

    w0 = tf.Print(w0, [w0], message = 'this is weight hiddenlayer_0', summarize=(w0.s[0]*w0.s[1]))
    w1 = tf.Print(w1, [w1], message = 'this is weight hiddenlayer_1', summarize=(w1.s[0]*w1.s[1]))
    w2 = tf.Print(w2, [w2], message = 'this is weight hiddenlayer_2', summarize=(w2.s[0]*w2.s[1]))
    wL = tf.Print(wL, [wL], message = 'this is weight logits', summarize=(wL.s[0]*wL.s[1]))

    #biases for layer 0, 1, 2 and logits
    b0 = graph.get_tensor_by_name("dnn/hiddenlayer_0/biases/part_0:0")
    b0.s = graph.get_tensor_by_name("dnn/hiddenlayer_0/biases/part_0:0").shape
    b1 = graph.get_tensor_by_name("dnn/hiddenlayer_1/biases/part_0:0")
    b1.s = graph.get_tensor_by_name("dnn/hiddenlayer_1/biases/part_0:0").shape
    b2 = graph.get_tensor_by_name("dnn/hiddenlayer_2/biases/part_0:0")
    b2.s = graph.get_tensor_by_name("dnn/hiddenlayer_2/biases/part_0:0").shape
    bL = graph.get_tensor_by_name("dnn/logits/biases/part_0:0")
    bL.s = graph.get_tensor_by_name("dnn/logits/biases/part_0:0").shape

    b0 = tf.Print(b0, [b0], message = 'this is bias hiddenlayer_0', summarize=(b0.s[0]))
    b1 = tf.Print(b1, [b1], message = 'this is bias hiddenlayer_1', summarize=(b1.s[0]))
    b2 = tf.Print(b2, [b2], message = 'this is bias hiddenlayer_2', summarize=(b2.s[0]))
    bL = tf.Print(bL, [bL], message = 'this is bias logits', summarize=(bL.s[0]))

    #tf.add(w0, 0).eval()
    #tf.add(w1, 0).eval()
    #tf.add(w2, 0).eval()
    #tf.add(wL, 0).eval()
    #tf.add(b0, 0).eval()
    #tf.add(b1, 0).eval()
    #tf.add(b2, 0).eval()
    #tf.add(bL, 0).eval()

if __name__ == "__main__":
  main()
