#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Moises Martinez & Nerea Luis
# Conference: Codemotion 2017
# Title: TensorFlow 101 - From theory to practice
# Email: momartinm@gmail.com
# Email: nerea.luis@gmail.com

from __future__ import print_function
import tensorflow as tf
import numpy

random_generator = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 300
display_step = 45

# Training Data (x: age: y: height)
train_X = numpy.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
train_Y = numpy.asarray([45.0, 54.3, 62.4, 65.9, 71.4, 82.3, 87.9, 91.1, 100.3, 105.3, 110.5, 125.4, 145.3, 161.3, 164.9, 170.4, 175.8, 180.9, 182.3, 185.4])
n_samples = train_X.shape[0]

# Test Data
test_X = numpy.asarray([5.0, 10.0, 11.0, 14.0, 18.0])
test_Y = numpy.asarray([72.2, 104.8, 111.8, 160.2, 179.8])

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model variables (y = b*x + W)
W = tf.Variable(random_generator.randn(), name="weight")
b = tf.Variable(random_generator.randn(), name="bias")

pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Start training phase
with tf.sessionion() as session:

    # Run the initializer
    session.run(tf.global_variables_initializer())

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            session.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = session.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", session.run(W), "b=", session.run(b))

    print("Optimization Finished!")
    training_cost = session.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", session.run(W), "b=", session.run(b), '\n')


    print("Testing model (Mean square loss Comparison)")
    testing_cost = session.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(training_cost - testing_cost))

    file_writer = tf.summary.FileWriter('/tmp/example2', session.graph)
