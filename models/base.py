from abc import abstractmethod

import json
import os
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm


class BaseModel:
    """
    Model base class for training and predicting
    """

    def __init__(self, winsize, sampling_rate,
                 learning_rate=0, epochs=100, batch_size=128, verbose=0):
        self.winsize = winsize
        self.sampling_rate = sampling_rate
        self.spw = self.sampling_rate * self.winsize
        self.model = None

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def summary(self):
        if self.model:
            self.model.summary()
        else:
            raise ValueError("The model has not yet been initialized.")

    def save_history(self, history, i, save_dir):
        json.dump(str(history.history), open(
            os.path.join(save_dir, f"history_{i}.json"), "w+"))

    def save_mae(self, mae, save_dir):
        np.savez(os.path.join(save_dir, f"mae_{self.winsize}"), mae)

    @abstractmethod
    def create_model(self):
        raise NotImplementedError

    @property
    def callbacks(self):
        return []

    def _reshape(self, arr, method, idx, shape):
        return method([np.array(arr[i]) for i in idx]).reshape(*shape)

    def _train_val_test_split(self, arr, method, train_idx, val_idx, test_idx, shape):
        train = self._reshape(arr, method, train_idx, shape)
        val = self._reshape(arr, method, val_idx, shape)
        test = self._reshape(arr, method, test_idx, shape)
        return train, val, test

    def iteration(self, x, y, train_indices, val_indices, test, rr=None):
        x_train, x_val, x_test = self._train_val_test_split(
            x,
            np.vstack,
            train_idx=train_indices,
            val_idx=val_indices,
            test_idx=[test],
            shape=(-1, self.spw, 1),
        )
        y_train, y_val, y_test = self._train_val_test_split(
            y,
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

        y_pred = self.model.predict(x_test, verbose=self.verbose)
        performance = np.absolute(y_pred - y_test)
        mae = np.average(performance)

        return mae, history

    def train(self, x, y, save_dir, rr=None):
        maes = []

        for test in tqdm(range(len(x)), desc="Leave-one-out evaluation"):
            persons = [i for i in range(len(x)) if i != test]
            kf = KFold(n_splits=5, shuffle=True)

            fold_mae = []

            for i, (train_indices, val_indices) in enumerate(kf.split(persons)):
                mae, history = self.iteration(
                    x, y, train_indices, val_indices, test, rr)

                fold_mae.append(mae)
                self.save_history(history, f"{test}_{i}", save_dir)

                del self.model
                self.create_model()

            maes.append(fold_mae)

        maes = np.array(maes)
        self.save_mae(maes, save_dir)
        self.print_results(maes)

    def print_results(self, maes):
        maes_by_subject = np.mean(maes, axis=1)
        print(f"Mean MAE = {np.mean(maes_by_subject):.3f}")
        print(f"Std Dev MAE = {np.std(maes_by_subject):.3f}")
