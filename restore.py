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
    w0_s = graph.get_tensor_by_name("dnn/hiddenlayer_0/weights/part_0:0").shape
    w1 = graph.get_tensor_by_name("dnn/hiddenlayer_1/weights/part_0:0")
    w1_s = graph.get_tensor_by_name("dnn/hiddenlayer_1/weights/part_0:0").shape
    w2 = graph.get_tensor_by_name("dnn/hiddenlayer_2/weights/part_0:0")
    w2_s = graph.get_tensor_by_name("dnn/hiddenlayer_2/weights/part_0:0").shape
    wL = graph.get_tensor_by_name("dnn/logits/weights/part_0:0")
    wL_s = graph.get_tensor_by_name("dnn/logits/weights/part_0:0").shape

    w0 = tf.Print(w0, [w0], message = 'this is weight hiddenlayer_0 size:{}'.format(w0_s), summarize=(w0_s[0]*w0_s[1]))
    w1 = tf.Print(w1, [w1], message = 'this is weight hiddenlayer_1 size:{}'.format(w1_s), summarize=(w1_s[0]*w1_s[1]))
    w2 = tf.Print(w2, [w2], message = 'this is weight hiddenlayer_2 size:{}'.format(w2_s), summarize=(w2_s[0]*w2_s[1]))
    wL = tf.Print(wL, [wL], message = 'this is weight logits size:{}'.format(wL_s), summarize=(wL_s[0]*wL_s[1]))

    #biases for layer 0, 1, 2 and logits
    b0 = graph.get_tensor_by_name("dnn/hiddenlayer_0/biases/part_0:0")
    b0_s = graph.get_tensor_by_name("dnn/hiddenlayer_0/biases/part_0:0").shape
    b1 = graph.get_tensor_by_name("dnn/hiddenlayer_1/biases/part_0:0")
    b1_s = graph.get_tensor_by_name("dnn/hiddenlayer_1/biases/part_0:0").shape
    b2 = graph.get_tensor_by_name("dnn/hiddenlayer_2/biases/part_0:0")
    b2_s = graph.get_tensor_by_name("dnn/hiddenlayer_2/biases/part_0:0").shape
    bL = graph.get_tensor_by_name("dnn/logits/biases/part_0:0")
    bL_s = graph.get_tensor_by_name("dnn/logits/biases/part_0:0").shape

    b0 = tf.Print(b0, [b0], message = 'this is bias hiddenlayer_0 size:{}'.format(b0_s), summarize=(b0_s[0]))
    b1 = tf.Print(b1, [b1], message = 'this is bias hiddenlayer_1 size:{}'.format(b1_s), summarize=(b1_s[0]))
    b2 = tf.Print(b2, [b2], message = 'this is bias hiddenlayer_2 size:{}'.format(b2_s), summarize=(b2_s[0]))
    bL = tf.Print(bL, [bL], message = 'this is bias logits size{}'.format(bL_s), summarize=(bL_s[0]))

    
    sess.run(w0)
    sess.run(w1)
    sess.run(w2)
    sess.run(wL)
    sess.run(b0)
    sess.run(b1)
    sess.run(b2)
    sess.run(bL)

if __name__ == "__main__":
  main()
