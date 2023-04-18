from .model_encoder import ModelEncoder

model_name = 'msmarco-distilbert-base-v3'
model = ModelEncoder(model_name, dims=768)


def encode(text, cached=True):
    return model.encode(text, cached=cached)
