import tensorflow as tf

#load in fashion MNIST data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#define an instantiate a custom callback

""" class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        modelAcu = logs.get('accuracy')
        print(f"Seen accuray -> {modelAcu}")
        if(epoch > 1 and modelAcu < 0.85): # < ---------------------------------------------------------  RESUME HERE! (last code block in collab)
            self.model.stop_training = True
callbacks = myCallback() """

#define the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])


training_images = training_images
test_images = test_images

#compile the model
model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#fit the model to the training data
model.fit(training_images, training_labels, epochs=5)

#test the model on the test data
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
