import os
import sys

import tensorflow as tf

from src import cifar10
from src import config


def setup():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if '.' not in sys.path:
        sys.path.insert(0, '')


def run():
    num_workers = len(config.tf_config['cluster']['worker'])
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    global_batch_size = config.per_worker_batch_size * num_workers

    multi_worker_dataset = cifar10.cifar10_dataset(global_batch_size)

    with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = cifar10.build_and_compile_cnn_model()

    multi_worker_model.fit(multi_worker_dataset, epochs=config.epochs, steps_per_epoch=config.steps_per_epoch)


if __name__ == '__main__':
    setup()
    run()
