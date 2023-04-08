from sentence_transformers import SentenceTransformer


model_name = 'ronanki/ml_use_512_MNR_15'
model = SentenceTransformer(model_name)


def encode(text):
    encoded = model.encode(text)
    if encoded.shape != (512,):
        import pdb; pdb.set_trace()
    assert encoded.shape == (512,)
    return encoded
