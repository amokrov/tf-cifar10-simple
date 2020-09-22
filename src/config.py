tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456', 'localhost:23457']
    },
    'task': {'type': 'worker', 'index': 0}
}
batch_size = 64
epochs = 2
steps_per_epoch = 782
per_worker_batch_size = 64
