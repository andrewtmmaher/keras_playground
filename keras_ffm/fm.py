import itertools
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, dot, add, Reshape

features = [1, 0.5], [2, 0], [3, 0], [4, 1000.0], [5, -12.0]


def build_factorization_machine(nb_features, latent_dimension):

    inputs = [Input((1,)) for __ in range(nb_features)]
    vectors = [
        Dense(latent_dimension, use_bias=False)(input) for input in inputs]

    dot_products = add([
        (dot([v1, v2], axes=1)) for v1, v2 in itertools.combinations(vectors, 2)])

    output = Dense(1, activation='sigmoid')(dot_products)

    model = Model(inputs=inputs, outputs=output)
    model.compile('adam', loss='binary_crossentropy')

    return model


fm = build_factorization_machine(5, 2)

print(fm.summary())
print(fm.predict(features))
