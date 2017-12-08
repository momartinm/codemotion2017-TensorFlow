#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Moises Martinez & Nerea Luis
# Conference: Codemotion 2017
# Title: TensorFlow 101 - From theory to practice
# Email: momartinm@gmail.com
# Email: nerea.luis@gmail.com

import tensorflow as tf
import numpy as np

session = tf.Session()

x = tf.placeholder("float", [1,4])
m = tf.Variable(tf.random_normal([4,4]), name='m')
y = tf.matmul(x, m)
r = tf.nn.relu(y)

session.run(tf.global_variables_initializer())

print(session.run(r, feed_dict = {x:np.array([[1.0,2.0,3.0,4.0]])}))

file_writer = tf.summary.FileWriter('/tmp/example1', session.graph)

session.close()
