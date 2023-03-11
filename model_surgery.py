def reconstruct_model(backbone, new_model):
    x = backbone.input
    d = dict()
    for layer in backbone.layers:
        d[layer.name] = layer
        if isinstance(layer.input, list):

