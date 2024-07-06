import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.activations import tanh
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
from adabelief_tf import AdaBeliefOptimizer

from models.base import BaseModel


class RRWaveNet(BaseModel):
    """
    Implementation of RRWaveNet (ours) as appear in

    P. Osathitporn, G. Sawadwuthikul, P. Thuwajit, K. Ueafuea, T. Mateepithaktham, N. Kunaseth,
    T. Choksatchawathi, P. Punyabukkana, E. Mignot, and T. Wilaiprasitporn,
    "RRWaveNet: A compact end-to-end multiscale residual cnn for robust ppg respiratory rate
    estimation," in IEEE Internet of Things Journal, vol. 10, no. 18, pp. 15943-15952, 2023.
    """

    def __init__(self, winsize, sampling_rate,
                 learning_rate=1e-3, epsilon=1e-14, epochs=69420, batch_size=100, verbose=0):
        super().__init__(winsize=winsize, sampling_rate=sampling_rate,
                         learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                         verbose=verbose)
        self.epsilon = epsilon
        self.create_model()

    def create_model(self):
        inputs = Input(shape=(self.spw, 1), name="input_layer")
        self.initializer = HeNormal()

        x = Concatenate()([
            self._multi_conv(inputs, 16),
            self._multi_conv(inputs, 32),
            self._multi_conv(inputs, 64)
        ])

        for ch in [64, 64, 128, 128, 256, 256, 512, 512]:
            x = self._resconv(ch, x)

        x = GlobalAveragePooling1D()(x)
        x = ReLU()(x)
        hp_intermediate = 64
        x = Dense(2 * hp_intermediate)(x)
        x = ReLU()(x)
        x = Dense(hp_intermediate)(x)
        x = tanh(x)
        x = Dense(1)(x)

        model = Model(inputs, x, name="RRWaveNet")
        optimizer = AdaBeliefOptimizer(
            learning_rate=self.learning_rate, epsilon=self.epsilon,
            rectify=True, print_change_log=False
        )
        model.compile(optimizer=optimizer, loss=MeanSquaredError())
        self.model = model

    def _multi_conv(self, x, kernel_size):
        x = Conv1D(1, kernel_size=kernel_size, strides=5,
                   padding="same", kernel_initializer=self.initializer)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling1D(pool_size=3, strides=2, padding="same")(x)
        return x

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
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=4),
        ]
