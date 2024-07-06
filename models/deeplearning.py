import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Add,
    Conv1D,
    Dense,
    Flatten,
    MaxPooling1D,
    ReLU,
)
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam

from models.base import BaseModel


class DeepLearning(BaseModel):
    """
    Implementation of the deep learning model from

    D. Bian, P. Mehta, and N. Selvaraj,
    "Respiratory rate estimation using PPG: A deep learning approach,"
    in Proc. 42nd Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC), 2020, pp. 5948-5952.
    """

    def __init__(self, winsize, sampling_rate,
                 learning_rate=1e-3, epochs=69420, batch_size=100, verbose=0):
        super().__init__(winsize=winsize, sampling_rate=sampling_rate,
                         learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                         verbose=verbose)
        self.create_model()

    def create_model(self):
        inputs = Input(shape=(self.spw, 1), name="input_layer")

        f = 6
        x = self._res(inputs, filter_size=f)
        for _ in range(4):
            f *= 2
            x = self._res(x, filter_size=f)

        x = MaxPooling1D(strides=2, padding="same")(x)
        x = Flatten()(x)
        x = Dense(20)(x)
        x = Dense(10)(x)
        x = Dense(1)(x)

        self.model = Model(inputs, x, name="DeepLearning")
        self.model.compile(optimizer=Adam(
            learning_rate=self.learning_rate), loss=MeanAbsoluteError())

    def _res(self, x, filter_size):
        x = Conv1D(filter_size, kernel_size=3, strides=2, padding="same")(x)
        a = x
        b = Conv1D(filter_size, kernel_size=3, strides=1, padding="same")(x)
        b = Conv1D(filter_size, kernel_size=3, strides=1, padding="same")(b)
        out = Add()([a, b])
        out = ReLU()(out)
        return out

    @property
    def callbacks(self):
        return [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)]
