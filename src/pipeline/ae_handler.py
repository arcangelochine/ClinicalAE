from datetime import datetime

from .models import AutoEncoder, MultiModalAutoEncoder

type AE = AutoEncoder or MultiModalAutoEncoder

class AutoEncoderHandler:
    def __init__(self, autoencoder: AE, x_train, y_train, verbose: int = 0):
        assert autoencoder is not None, 'Autoencoder must be provided'
        self.autoencoder = autoencoder
        self.x_train = x_train
        self.y_train = y_train
        self.verbose = verbose

    def train(self, epochs=50, batch_size=2, shuffle=True, validation_split=0.2, callbacks=None):
        start = datetime.now()
        history = self.autoencoder.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                                       shuffle=shuffle, validation_split=validation_split, verbose=self.verbose,
                                       callbacks=callbacks)
        end = datetime.now()

        return history, end - start

    def encode(self, inputs):
        if isinstance(self.autoencoder, MultiModalAutoEncoder):
            return self.autoencoder.encode(inputs).numpy()
        return self.autoencoder.encode(inputs)


__all__ = ['AutoEncoderHandler', 'AE']
