import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class AEBlock(tf.keras.layers.Layer):
    def __init__(self, units: int, drop_rate: float, seed: int = 42):
        assert 0 < units, 'Invalid units'
        assert 0 <= drop_rate <= 1, 'Invalid dropout rate'

        super(AEBlock, self).__init__()

        self.units = units
        self.drop_rate = drop_rate
        self.seed = seed

        self.kernel_init = tf.keras.initializers.GlorotNormal(seed=self.seed)
        self.bias_init = tf.keras.initializers.Zeros()

    def build(self):
        self.dense = tf.keras.layers.Dense(self.units, activation='relu', kernel_initializer=self.kernel_init,
                                           bias_initializer=self.bias_init)
        self.bn = tf.keras.layers.BatchNormalization()
        self.drop = tf.keras.layers.Dropout(self.drop_rate) if self.drop_rate > 0 else None

    def call(self, inputs, training=False):
        x = self.dense(inputs)
        x = self.bn(x, training=training)
        return self.drop(x, training=training) if self.drop else x

    def get_config(self):
        return {
            'units': self.units,
            'drop_rate': self.drop_rate,
            'seed': self.seed
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class AutoEncoder(tf.keras.Model):
    def __init__(self, input_size: int, encoder_units: list[int], encoder_activation: str = 'tanh',
                 decoder_activation: str = 'linear', drop_rate: float = 0, seed: int = 42):
        super(AutoEncoder, self).__init__()
        assert 0 < input_size, 'Invalid input size'
        assert 0 < len(encoder_units), 'No encoder units provided'
        assert 0 < min(encoder_units), 'Invalid encoder units'
        assert 0 <= drop_rate <= 1, 'Invalid dropout rate'

        self.input_size = input_size
        self.encoder_units = encoder_units
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.drop_rate = drop_rate
        self.seed = seed

    def build(self):
        latent_units = self.encoder_units[-1]

        self.encoder = tf.keras.models.Sequential(name='encoder')
        self.encoder.add(tf.keras.layers.InputLayer(shape=(self.input_size,)))

        for units in self.encoder_units[:-1]:
            self.encoder.add(AEBlock(units, self.drop_rate, self.seed))
        self.encoder.add(tf.keras.layers.Dense(latent_units, activation=self.encoder_activation))

        self.decoder = tf.keras.models.Sequential(name='decoder')
        self.decoder.add(tf.keras.layers.InputLayer(shape=(latent_units,)))

        for units in reversed(self.encoder_units[:-1]):
            self.decoder.add(AEBlock(units, 0, self.seed))
        self.decoder.add(tf.keras.layers.Dense(self.input_size, activation=self.decoder_activation))

    def call(self, inputs, training=False):
        return self.decoder(self.encoder(inputs, training=training), training=training)

    def encode(self, inputs):
        return self.encoder.predict(inputs)

    def get_config(self):
        return {
            'input_size': self.input_size,
            'encoder_units': self.encoder_units,
            'encoder_activation': self.encoder_activation,
            'decoder_activation': self.decoder_activation,
            'drop_rate': self.drop_rate,
            'seed': self.seed
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


__all__ = ['AutoEncoder']
