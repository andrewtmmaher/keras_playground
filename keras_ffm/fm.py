"""Keras implementation of Factorization Machine (FM)."""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Input, dot, add, Reshape, multiply, subtract, Concatenate)

features = [1, 0.5], [2, 0], [3, 0], [4, 1000.0], [5, -12.0]


def build_factorization_machine(nb_features, latent_dimension):

    print('Defining inputs')
    inputs = [Input((1,)) for __ in range(nb_features)]

    print('Defining dense layers')
    vectors = [
        Dense(latent_dimension, use_bias=False)(input) for input in inputs]

    square_of_sums = tf.square(tf.reduce_sum(vectors, axis=1, keep_dims=True))
    vectors = Concatenate()(vectors)
    sum_of_squares = tf.reduce_sum(multiply([vectors, vectors]), axis=1, keep_dims=True)
    dot_product_sum = tf.reduce_sum(square_of_sums - sum_of_squares, axis=2, keep_dims=False)

    print('Defining output')
    output = tf.sigmoid(dot_product_sum)

    model = Model(inputs=inputs, outputs=output)
    model.compile('adam', loss='binary_crossentropy')

    print(model.summary())

    return model
