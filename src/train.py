import iresnet

import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nets', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

NUM_CLASSES = 100
INPUT_SHAPE = (32, 32, 3)
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

cifar100 = tf.keras.datasets.cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(BATCH_SIZE)

if args.nets == 'iresnet18':
    get_model = iresnet.iresnet18
elif args.nets == 'iresnet34':
    get_model = iresnet.iresnet34
elif args.nets == 'iresnet50':
    get_model = iresnet.iresnet50
elif args.nets == 'iresnet101':
    get_model = iresnet.iresnet101
elif args.nets == 'iresnet152':
    get_model = iresnet.iresnet152
elif args.nets == 'iresnet200':
    get_model = iresnet.iresnet200
elif args.nets == 'iresnet302':
    get_model = iresnet.iresnet302
elif args.nets == 'iresnet404':
    get_model = iresnet.iresnet404
elif args.nets == 'iresnet1001':
    get_model = iresnet.iresnet1001
else:
    raise NotImplementedError

model = get_model(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'EPOCH: {}, Loss: {}, Accuracy: {}, TestLoss: {}, TestAccuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))
