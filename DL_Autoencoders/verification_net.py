import numpy as np
from tensorflow import keras
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

from stacked_mnist_tf import DataMode, StackedMNISTData


class VerificationNet:
    def __init__(
        self, force_learn: bool = False, file_name: str = "C:/Projects/public/DL_Autoencoders/models/RGB_verification_model.weights.h5"
    ) -> None:
        """
        Define model and set some parameters.
        The model is  made for classifying one channel only -- if we are looking at a
        more-channel image we will simply do the thing one-channel-at-the-time.
        """
        self.force_relearn = force_learn
        self.file_name = file_name

        model = Sequential()
        model.add(
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1))
        )
        for _ in range(3):
            model.add(Conv2D(64, (3, 3), activation="relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation="softmax"))

        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            metrics=["accuracy"],
        )

        self.model = model
        self.done_training = self.load_weights()

    def load_weights(self):
        # noinspection PyBroadException
        try:
            self.model.load_weights(filepath=self.file_name)
            # print(f"Read model from file, so I do not retrain")
            done_training = True

        except:
            print(
                f"Could not read weights for verification_net from file. Must retrain..."
            )
            done_training = False

        return done_training

    def train(self, generator: StackedMNISTData, epochs: int = 10) -> bool:
        """
        Train model if required. As we have a one-channel model we take care to
        only use the first channel of the data.
        """
        self.done_training = self.load_weights()

        if self.force_relearn or self.done_training is False:
            # Get hold of data
            x_train, y_train = generator.get_full_data_set(training=True)
            x_test, y_test = generator.get_full_data_set(training=False)

            # "Translate": Only look at "red" channel; only use the last digit. Use one-hot for labels during training
            x_train = x_train[:, :, :, [0]]
            y_train = keras.utils.to_categorical((y_train % 10).astype(int), 10)
            x_test = x_test[:, :, :, [0]]
            y_test = keras.utils.to_categorical((y_test % 10).astype(int), 10)

            # Fit model
            self.model.fit(
                x=x_train,
                y=y_train,
                batch_size=1024,
                epochs=epochs,
                validation_data=(x_test, y_test),
            )

            # Save weights and leave
            self.model.save_weights(filepath=self.file_name)
            self.done_training = True

        return self.done_training

    def predict(self, data: np.ndarray) -> tuple:
        """
        Predict the classes of some specific data-set. This is basically prediction using keras, but
        this method is supporting multi-channel inputs.
        Since the model is defined for one-channel inputs, we will here do one channel at the time.

        The rule here is that channel 0 define the "ones", channel 1 defines the tens, and channel 2
        defines the hundreds.

        Since we later need to know what the "strength of conviction" for each class-assessment we will
        return both classifications and the belief of the class.
        For multi-channel images, the belief is simply defined as the probability of the allocated class
        for each channel, multiplied.
        """
        num_channels = data.shape[-1]

        if self.done_training is False:
            # Model is not trained yet...
            raise ValueError("Model is not trained, so makes no sense to try to use it")

        predictions = np.zeros((data.shape[0],))
        beliefs = np.ones((data.shape[0],))
        for channel in range(num_channels):
            channel_prediction = self.model.predict(data[:, :, :, [channel]])
            beliefs = np.multiply(beliefs, np.max(channel_prediction, axis=1))
            predictions += np.argmax(channel_prediction, axis=1) * np.power(10, channel)

        return predictions, beliefs

    def check_class_coverage(
        self, data: np.ndarray, tolerance: float = 0.8
    ) -> float:
        """
        Out of the total number of classes that can be generated, how many are in the data-set?
        I'll only count samples for which the network asserts there is at least tolerance probability
        for a given class.
        """
        num_classes_available = np.power(10, data.shape[-1])
        predictions, beliefs = self.predict(data=data)

        # Only keep predictions where all channels were legal
        predictions = predictions[beliefs >= tolerance]

        # Coverage: Fraction of possible classes that were seen
        coverage = float(len(np.unique(predictions))) / num_classes_available
        return coverage

    def check_predictability(
        self, data: np.ndarray, correct_labels: list = None, tolerance: float = 0.8
    ) -> tuple:
        """
        Out of the number of data points retrieved, how many are we able to make predictions about?
        ... and do we guess right??

        Inputs here are
        - data samples -- size (N, 28, 28, color-channels)
        - correct labels -- if we have them. List of N integers
        - tolerance: Minimum level of "confidence" for us to make a guess

        """
        # Get predictions; only keep those where all channels were "confident enough"
        predictions, beliefs = self.predict(data=data)
        predictions = predictions[beliefs >= tolerance]
        predictability = len(predictions) / len(data)

        if correct_labels is not None:
            # Drop those that were below threshold
            correct_labels = correct_labels[beliefs >= tolerance]
            accuracy = np.sum(predictions == correct_labels) / len(data)
        else:
            accuracy = None

        return predictability, accuracy


if __name__ == "__main__":
    gen = StackedMNISTData(mode=DataMode.COLOR_BINARY_MISSING, default_batch_size=25000*4)
    net = VerificationNet(force_learn=False,
        file_name = "C:/Projects/public/DL_Autoencoders/models/net_COLOR_BINARY_MISSING.weights.h5")
    net.train(generator=gen, epochs=10)  # was originally 5

    # I have no data generator (VAE or whatever) here, so just use a sampled set
    img, labels = gen.get_random_batch(training=True, batch_size=25000)
    cov = net.check_class_coverage(data=img, tolerance=0.98)
    pred, acc = net.check_predictability(data=img, correct_labels=labels)
    print(f"Coverage: {100*cov:.2f}%")
    print(f"Predictability: {100*pred:.2f}%")
    print(f"Accuracy: {100 * acc:.2f}%")

    img, labels = gen.get_random_batch(training=True, batch_size=5)
    predictedLabels = net.predict(data=img)
    print(f"Predicted labels: {predictedLabels}")
    print(f"Correct labels: {labels}")

    