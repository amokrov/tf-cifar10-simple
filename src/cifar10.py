import tensorflow as tf


def cifar10_dataset(batch_size):
    c_train, c_test = tf.keras.datasets.cifar10.load_data()
    buffer_size = len(c_train[0])

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255

        return image, label

    train_dataset = tf.data.Dataset.from_tensor_slices(
        scale(*c_train)).shuffle(buffer_size).repeat().batch(batch_size)
    eval_dataset = tf.data.Dataset.from_tensor_slices(
        scale(*c_test)).batch(batch_size)

    return train_dataset, eval_dataset


def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model
