import tensorflow as tf
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Conv1DTranspose,
    LeakyReLU,
    ReLU,
)
from scipy.signal import find_peaks

from models.base import BaseModel


class RespNet(BaseModel):
    def __init__(self, winsize, sampling_rate,
                 learning_rate=0.01, momentum=0.7, epochs=69420, batch_size=100, verbose=0):
        super().__init__(winsize=winsize, sampling_rate=sampling_rate,
                         learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                         verbose=verbose)
        self.momentum = momentum
        self.create_model()

    def create_model(self):
        inputs = Input(shape=(self.spw, 1), name="input_layer")

        enc1 = self._inception(inputs, 1, 1)
        cov1 = self._conv(enc1, 8)

        enc2 = self._inception(cov1, 8, 2)
        cov2 = self._conv(enc2, 16)

        enc3 = self._inception(cov2, 16, 4)
        cov3 = self._conv(enc3, 32)

        enc4 = self._inception(cov3, 32, 8)
        cov4 = self._conv(enc4, 64)

        enc5 = self._inception(cov4, 64, 16)
        cov5 = self._conv(enc5, 128)

        enc6 = self._inception(cov5, 128, 32)
        cov6 = self._conv(enc6, 256)

        enc7 = self._inception(cov6, 256, 64)
        cov7 = self._conv(enc7, 512)

        enc8 = self._inception(cov7, 512, 128)

        dcv1 = self._deconv(enc8, 256)
        cat1 = Concatenate()([dcv1, enc7])
        dec1 = self._inception(cat1, 512, 128)

        dcv2 = self._deconv(dec1, 128)
        cat2 = Concatenate()([dcv2, enc6])
        dec2 = self._inception(cat2, 256, 64)

        dcv3 = self._deconv(dec2, 64)
        cat3 = Concatenate()([dcv3, enc5])
        dec3 = self._inception(cat3, 128, 32)

        dcv4 = self._deconv(dec3, 32)
        cat4 = Concatenate()([dcv4, enc4])
        dec4 = self._inception(cat4, 64, 16)

        dcv5 = self._deconv(dec4, 16)
        cat5 = Concatenate()([dcv5, enc3])
        dec5 = self._inception(cat5, 32, 8)

        dcv6 = self._deconv(dec5, 8)
        cat6 = Concatenate()([dcv6, enc2])
        dec6 = self._inception(cat6, 16, 4)

        dcv7 = self._deconv(dec6, 4)
        cat7 = Concatenate()([dcv7, enc1])
        dec7 = self._inception(cat7, 8, 2)

        x = self._map(dec7, 1)

        model = Model(inputs, x, name="RespNet")
        model.compile(optimizer=SGD(
            learning_rate=self.learning_rate, momentum=self.momentum), loss=Huber())
        self.model = model

    def _inception(self, x, filter_in, filter_out):
        a = Conv1D(filter_out, kernel_size=1)(x)
        a = BatchNormalization()(a)

        b = self._dilated(x, filter_in, filter_out, dilation=2)
        c = self._dilated(x, filter_in, filter_out, dilation=4)
        d = self._dilated(x, filter_in, filter_out, dilation=4)

        out = Concatenate()([a, b, c, d])
        out = Add()([out, x])
        return out

    def _dilated(self, x, filter_in, filter_out, dilation):
        x = Conv1D(filter_in, kernel_size=1)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv1D(filter_out, kernel_size=3,
                   padding="same", dilation_rate=dilation)(x)
        x = BatchNormalization()(x)
        return x

    def _conv(self, x, filter_size, strides=2, kernel_size=4):
        x = Conv1D(filter_size, padding="same",
                   kernel_size=kernel_size, strides=strides)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        return x

    def _map(self, x, filter_size):
        x = Conv1D(filter_size, kernel_size=1)(x)
        return x

    def _deconv(self, x, filter_size, strides=2, kernel_size=4):
        x = Conv1DTranspose(filter_size, padding="same",
                            kernel_size=kernel_size, strides=strides)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        return x

    @property
    def callbacks(self):
        return [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=4),
        ]

    def _compute_rr(self, y, distance):
        all_cal_rr = []

        for yp in y:
            yp = yp.numpy().reshape(-1)
            peaks, _ = find_peaks(yp, distance=distance)
            num_peak = len(peaks)

            cal = self.winsize / num_peak
            cal_rr = 60 / cal
            all_cal_rr.append(cal_rr)

        return all_cal_rr

    def _grid_search(self, y, rr):
        best_mae = None
        for distance in range(300, 1001, 50):
            rr_pred = self._compute_rr(y, distance=distance)
            performance = np.absolute(rr_pred - rr)
            mae = np.average(performance)

            if best_mae is None or mae < best_mae:
                best_mae = mae
                best_distance = distance

        return best_distance

    def iteration(self, x, y, train_indices, val_indices, test, rr=None):
        x_train, x_val, x_test = self._train_val_test_split(
            x,
            np.vstack,
            train_idx=train_indices,
            val_idx=val_indices,
            test_idx=[test],
            shape=(-1, self.spw, 1),
        )
        y_train, y_val, _ = self._train_val_test_split(
            y,
            np.hstack,
            train_idx=train_indices,
            val_idx=val_indices,
            test_idx=[test],
            shape=(-1, 1),
        )
        rr_train, _, rr_test = self._train_val_test_split(
            rr,
            np.hstack,
            train_idx=train_indices,
            val_idx=val_indices,
            test_idx=[test],
            shape=(-1, 1),
        )

        history = self.model.fit(
            x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
            validation_data=(x_val, y_val), callbacks=self.callbacks, verbose=self.verbose
        )

        y_train_pred = self.model.predict(x_train)
        best_distance = self._grid_search(y=y_train_pred, rr=rr_train)

        y_pred = self.model.predict(x_test)
        rr_pred = self._compute_rr(y=y_pred, distance=best_distance)
        performance = np.absolute(rr_pred - rr_test)
        mae = np.average(performance)

        return mae, history
