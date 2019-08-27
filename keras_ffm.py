"""
Untested definition of Keras FFM implementation.

Almost certainly filled with many bugs! But the general idea for how to write
FFM in Keras should (largely) be correct.
"""
from tensorflow.keras.layers import Input, Dense, Reshape, Embedding, multiply, add


def _reshape_embedding(input, embedding, name):
    return Reshape((embedding.output_dim, 1), name=name)(embedding(input))


EMBEDDING_DIMENSION = 8

fields = [...]  # List of objects having attributes: id, name and nb_features

input_fields = {
    f.id: Input((1,), name=f.name) for f in fields}

embeddings = {}
for f1 in fields:
    for f2 in fields:
        if f2.id < f1.id:
            continue

        embeddings[(f1.id, f2.id)] = Embedding(
            f1.nb_features,
            EMBEDDING_DIMENSION,
            input_length=1
        )


products = []
for f1 in fields:
    for f2 in fields:
        if f2.id < f1.id:
            continue

    input_field_1 = _reshape_embedding(
        input_fields[f1.id],
        embeddings[(f1.id, f2.id)],
    )

    embedded_input_field_1 = embeddings[(f1.id, f2.id)](input_fields[f1.id])
    embedded_input_field_2 = embeddings[(f2.id, f1.id)](input_fields[f2.id])

    embedded_input_field_1 = \
        Reshape((embeddings[(f1.id, f2.id)].output_dim, 1))(embedded_input_field_1)
    embedded_input_field_2 = \
        Reshape((embeddings[(f1.id, f2.id)].output_dim, 1))(embedded_input_field_1)

    product = multiply([embedded_input_field_1, embedded_input_field_2], axes=1, normalize=False)
    products.append(Reshape((1,))(product))


added = add(products)

output = Dense(1, activation='sigmoid', name='sigmoid_activation')(added)
