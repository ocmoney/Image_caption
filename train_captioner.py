optimizer = optim.Adam([
    # Group 1: Frozen Embeddings (indices 0 to num_to_freeze-1)
    {'params': embedding_layer.weight[freeze_indices], 'lr': 0.0},
    # Group 2: Trainable Embeddings (indices num_to_freeze to end)
    {'params': embedding_layer.weight[train_indices], 'lr': 1e-3},
    # Add other model parameters here if needed
    # {'params': other_model_parameters, 'lr': 1e-3}
])