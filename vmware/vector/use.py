from .model_encoder import ModelEncoder


model_name = 'ronanki/ml_use_512_MNR_15'
model = ModelEncoder(model_name, dims=512)


def encode(text, cached=False):
    encoded = model.encode(text, cached=cached)
    assert encoded.shape == (512,)
    return encoded
