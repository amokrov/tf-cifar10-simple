import json
import os
import sys

from concurrent.futures import ProcessPoolExecutor

import tensorflow as tf

from src import cifar10
from src import config


def setup():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if '.' not in sys.path:
        sys.path.insert(0, '')


def cleanup():
    os.environ.pop('TF_CONFIG', None)


def run(worker_index=0):
    config.tf_config['task']['index'] = worker_index
    os.environ['TF_CONFIG'] = json.dumps(config.tf_config)
    num_workers = len(config.tf_config['cluster']['worker'])
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    global_batch_size = config.per_worker_batch_size * num_workers
    train_dataset, eval_dataset = cifar10.cifar10_dataset(global_batch_size)

    with strategy.scope():
        model = cifar10.get_model()

    model.fit(train_dataset,
              epochs=config.epochs,
              steps_per_epoch=config.steps_per_epoch,
              validation_data=eval_dataset)


if __name__ == '__main__':
    setup()
    num_workers = len(config.tf_config['cluster']['worker'])
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(run, range(num_workers))
    cleanup()
