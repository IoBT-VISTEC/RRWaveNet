import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    GlobalAveragePooling1D,
    MaxPooling1D,
    ReLU,
)
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import SGD

from models.base import BaseModel


class RespWatch(BaseModel):
    """
    Implementation of RespWatch model from

    R. Dai, C. Lu, M. Avidan, and T. Kannampallil,
    "RespWatch: Robust measurement of respiratory rate on smartwatches with photoplethysmography,"
    in Proc. Int. Conf. Internet Things Des. Implement., 2021, pp. 208-220.
    """

    def __init__(self, winsize, sampling_rate,
                 learning_rate=1e-3, epochs=69420, batch_size=100, verbose=0):
        super().__init__(winsize=winsize, sampling_rate=sampling_rate,
                         learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                         verbose=verbose)
        self.create_model()

    def create_model(self):
        inputs = Input(shape=(self.spw, 1), name="input_layer")
        self.initializer = HeNormal()

        x = Conv1D(1, kernel_size=100, strides=5, padding="same",
                   kernel_initializer=self.initializer)(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling1D(pool_size=3, strides=2)(x)

        for ch in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
            x = self._resconv(ch, x)

        x = GlobalAveragePooling1D()(x)
        x = ReLU()(x)
        hp_intermediate = 64
        x = Dense(hp_intermediate)(x)
        x = ReLU()(x)
        x = Dense(1)(x)

        model = Model(inputs, x, name="RespWatch")
        model.compile(optimizer=SGD(learning_rate=self.learning_rate), loss=MeanSquaredError())
        self.model = model

    def _resconv(self, N, x):
        y = x
        x = Conv1D(N, kernel_size=3, strides=1, padding="same",
                   kernel_initializer=self.initializer)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv1D(N, kernel_size=3, strides=1, padding="same",
                   kernel_initializer=self.initializer)(x)
        x = BatchNormalization()(x)
        x = Concatenate()([y, x])
        return x

    @property
    def callbacks(self):
        return [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.25, patience=4),
        ]
