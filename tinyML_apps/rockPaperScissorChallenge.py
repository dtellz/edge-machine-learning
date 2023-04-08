import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tqdm import tqdm
import pathlib

# Load the data


def format_image(image, label):
    image = tf.image.resize(image, (224, 224)) / 255.0
    return image, label


(raw_train, raw_validation, raw_test), metadata = tfds.load('rock_paper_scissors', split=[
    'train[:80%]', 'train[80%:90%]', 'train[90%:]'], with_info=True, as_supervised=True)

num_examples = metadata.splits['train'].num_examples
num_classes = metadata.features['label'].num_classes

BATCH_SIZE = 32
train_batches = raw_train.shuffle(
    num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = raw_validation.map(
    format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = raw_test.map(format_image).batch(1)

for image_batch, label_batch in train_batches.take(1):
    pass


module_selection = ("mobilenet_v2", 224, 1280)
handle_base, pixels, FV_SIZE = module_selection
