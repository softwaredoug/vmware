from .model_encoder import ModelEncoder

model_name = 'all-mpnet-base-v2'
model = ModelEncoder(model_name, dims=768)


def encode(text, cached=False):
    return model.encode(text, cached=cached)
