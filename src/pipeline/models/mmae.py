import tensorflow as tf
from .ae import AutoEncoder


@tf.keras.utils.register_keras_serializable()
class MultiModalAutoEncoder(tf.keras.Model):
    def __init__(self, autoencoders: list[AutoEncoder], seed: int = 42):
        super(MultiModalAutoEncoder, self).__init__()
        assert 0 < len(autoencoders), 'No autoencoder provided'

        self.seed = seed

        for autoencoder in autoencoders:
            autoencoder.seed = self.seed

        self.autoencoders = autoencoders

    def compile(self, optimizer, loss, loss_weights: list[float] = None):
        if loss_weights:
            assert len(loss_weights) == len(self.autoencoders), 'Loss weights AutoEncoders mismatch'

        self.optimizer = optimizer
        self.loss = loss
        self.loss_weights = loss_weights or [1 / len(self.autoencoders)] * len(self.autoencoders)
        super(MultiModalAutoEncoder, self).compile(optimizer=self.optimizer, loss=self.loss,
                                                   loss_weights=self.loss_weights)

    def build(self):
        for autoencoder in self.autoencoders:
            autoencoder.build()

    def call(self, inputs, training=False):
        assert len(inputs) == len(self.autoencoders), 'Input AutoEncoders mismatch'

        return [autoencoder.call(inputs=input_data, training=training) for autoencoder, input_data in
                zip(self.autoencoders, inputs)]

    def encode(self, inputs):
        encoded = [autoencoder.encode(inputs=input_data) for autoencoder, input_data in
                   zip(self.autoencoders, inputs)]

        return tf.concat(encoded, axis=1)

    def get_config(self):
        return {
            'autoencoders': [autoencoder.get_config() for autoencoder in self.autoencoders],
            'seed': self.seed
        }

    @classmethod
    def from_config(cls, config):
        autoencoders_config = config['autoencoders']
        autoencoders = [AutoEncoder.from_config(cfg) for cfg in autoencoders_config]
        return cls(autoencoders=autoencoders, seed=config['seed'])

    def get_compile_config(self):
        return {
            'optimizer': tf.keras.utils.serialize_keras_object(self.optimizer),
            'loss': self.loss,
            'loss_weights': self.loss_weights,
        }

    def compile_from_config(self, config):
        optimizer = tf.keras.utils.deserialize_keras_object(config['optimizer'])
        loss = tf.keras.utils.deserialize_keras_object(config['loss'])
        loss_weights = tf.keras.utils.deserialize_keras_object(config['loss_weights'])

        self.compile(optimizer, loss, loss_weights)


__all__ = ['MultiModalAutoEncoder']
