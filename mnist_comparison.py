import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa


# pipeline taken from https://www.tensorflow.org/datasets/keras_example
from one_cycle_adamw import OneCycleAdamW


def train_mnist(optimiser, experiment_title):
    batch_size = 128
    epochs = 20

    log_dir = f'logs/{experiment_title}/'
    summary_writer = tf.summary.create_file_writer(log_dir)

    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def model_loss(input, labels):
        pred = model(input)
        loss = loss_fn(labels, pred)
        return loss

    def train_model_step(input, labels):
        with tf.GradientTape() as tape:
            loss = model_loss(input, labels)

        trainables = model.trainable_weights
        grads = tape.gradient(loss, trainables)
        optimiser.apply_gradients(zip(grads, trainables))
        return loss

    step = 0
    for epoch in range(epochs):
        for train_step, (image, label) in enumerate(ds_train):
            loss = train_model_step(image, label)
            step += 1

            with summary_writer.as_default():
                tf.summary.scalar('train/model loss', loss, step=step)
                tf.summary.scalar('Train/Learning Rate', optimiser._decayed_lr(tf.float32), step=step)

        val_loss_sum = 0

        for val_step, (image, label) in enumerate(ds_test):
            val_model_loss = model_loss(image, label)
            val_loss_sum += val_model_loss

        with summary_writer.as_default():
            tf.summary.scalar('val/model loss', val_loss_sum / val_step, step=step)


if __name__ == '__main__':
    high_lr = 0.003
    high_wd = 0.0003

    low_lr = 0.001
    low_wd = 0.0001

    high_adam = tf.keras.optimizers.Adam(high_lr)
    train_mnist(high_adam, f'adam {high_lr} LR')

    low_adam = tf.keras.optimizers.Adam(low_lr)
    train_mnist(low_adam, f'adam {low_lr} LR')

    high_adamW = tfa.optimizers.AdamW(high_wd, high_lr)
    train_mnist(high_adam, f'adamW {high_lr} LR')

    low_adamW = tfa.optimizers.AdamW(low_wd, low_lr)
    train_mnist(low_adam, f'adamW {low_lr} LR')

    high_oc_adamw = OneCycleAdamW(high_lr, high_wd, 7000)
    train_mnist(high_oc_adamw, f'OC adamW {high_lr} LR')

    low_ic_adamw_ = OneCycleAdamW(low_lr, low_wd, 7000)
    train_mnist(low_ic_adamw_, f'OC adamW {low_lr} LR')
