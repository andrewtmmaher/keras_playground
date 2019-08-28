"""
Untested definition of Keras FFM implementation.

Almost certainly filled with many bugs! But the general idea for how to write
FFM in Keras should (largely) be correct.
"""
from tensorflow.keras.layers import Input, Dense, Reshape, Embedding, multiply, add


def _reshape_embedding(input, embedding, name):
    return Reshape((embedding.output_dim, 1), name=name)(embedding(input))


EMBEDDING_DIMENSION = 8

features = [...]  # List of objects having attributes: feature_id and field_id

input_features = {
    f.feature_id: Input((1,)) for f in fields}

embeddings = {}
for f1 in features:
    for f2 in features:
        if f1.field_id == f2.field_id:
            continue

        embeddings[(f1.feature_id, f2.field_id)] = Embedding(
            f1.nb_features,
            EMBEDDING_DIMENSION,
            input_length=1
        )


products = []
for f1 in fields:
    for f2 in fields:
        if f2.field_id == f1.field_id:
            continue

    embedded_input_feature_1 = embeddings[(f1.feature_id, f2.field_id)](input_fields[f1.feature_id])
    embedded_input_feature_2 = embeddings[(f2.feature_id, f1.field_id)](input_fields[f2.feature_id])

    embedded_input_feature_1 = \
        Reshape((embeddings[(f1.feature_id, f2.field_id)].output_dim, 1))(embedded_input_feature_1)
    embedded_input_feature_2 = \
        Reshape((embeddings[(f2.feature_id, f1.field_id)].output_dim, 1))(embedded_input_feature_2)

    product = multiply([embedded_input_feature_1, embedded_input_feature_2], axes=1, normalize=False)
    products.append(Reshape((1,))(product))


added = add(products)

output = Dense(1, activation='sigmoid', name='sigmoid_activation')(added)
