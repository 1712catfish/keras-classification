def classification_model_transfer_weight(model, m):
    m_backbone = m.layers[1]

    layers_loaded = []
    layers_not_avail = []

    j = 0
    for layer in model.layers[1].layers:
        while type(m_backbone.layers[j]) != type(layer):
            # print(f"Layer weight not available: {m_backbone.layers[j].name}")
            layers_not_avail.append(m_backbone.layers[j])
            j += 1

        # print(f"Load {layer.name} >> {m_backbone.layers[j].name}")
        layers_loaded.append(m_backbone.layers[j])
        m_backbone.layers[j].set_weights(layer.get_weights())
        j += 1

    print("Feature extractor:")
    print(f"Load {len(layers_loaded)} of {len(model.layers[1].layers)} total layers")
    print(f"{len(layers_not_avail)} layers not available\n")

    print("Classifier:")
    for layer, m_layer in zip(model.layers[2:], m.layers[2:]):
        m_layer.set_weights(layer.get_weights())
        print(f"Loaded {layer.name} >> {m_layer.name}")

    return m