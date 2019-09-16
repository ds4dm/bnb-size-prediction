import pickle
import tensorflow as tf
import tensorflow.keras as K


SEQUENCE_LENGTH = 50
FEATURE_SIZE = 40


class BaseModel(K.Model):
    """
    Our base model class, which implements basic save/restore.
    """
    def save_state(self, path):
        with open(path, 'wb') as f:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                pickle.dump(v.numpy(), f)

    def restore_state(self, path):
        with open(path, 'rb') as f:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                v.assign(pickle.load(f))


class Model(BaseModel):
    def __init__(self, feature_means=None, feature_stds=None):
        super().__init__()
        if feature_means is None:
            feature_means = tf.zeros(FEATURE_SIZE)
        if feature_stds is None:
            feature_stds = tf.ones(FEATURE_SIZE)
        self.norm = K.layers.Dense(feature_means.shape[0], trainable=False,
            kernel_initializer=lambda shape, dtype: tf.diag(1/feature_stds),
            bias_initializer=lambda shape, dtype: -feature_means/feature_stds)
        self.conv = K.Sequential([
            K.layers.Conv1D(filters=64, kernel_size=5, padding="same", activation='relu', 
                            kernel_initializer=K.initializers.Orthogonal(),
                            input_shape=(SEQUENCE_LENGTH, FEATURE_SIZE)),
            K.layers.MaxPool1D(pool_size=4, strides=2, padding='same'),
            K.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu',
                            kernel_initializer=K.initializers.Orthogonal()),
            K.layers.MaxPool1D(pool_size=4, strides=2, padding='same'),
            K.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu',
                            kernel_initializer=K.initializers.Orthogonal()),
            K.layers.MaxPool1D(pool_size=4, strides=2, padding='same'),
            K.layers.BatchNormalization(),
            K.layers.Flatten()
            ])
        self.dense = K.Sequential([
            K.layers.Dense(units=64, activation='relu', 
                           kernel_initializer=K.initializers.Orthogonal(),
                           input_shape=(448,)),
            K.layers.Dense(units=64, activation='relu',
                           kernel_initializer=K.initializers.Orthogonal()),
            K.layers.Dense(units=1,
                           kernel_initializer=K.initializers.Orthogonal())
            ])

        # build model right-away
        self.build((None, SEQUENCE_LENGTH, FEATURE_SIZE))

        # save / restore fix
        self.variables_topological_order = [v.name for v in self.variables]

    def build(self, input_shapes):
        _, sequence_length, feature_size = input_shapes

        if not self.built:
            self.norm.build(feature_size)
            self.conv.build((None, sequence_length, feature_size))
            self.dense.build((None, 448))
            self.built = True

    def call(self, inputs):
        inputs = self.norm(inputs)
        hidden = self.conv(inputs)
        output = self.dense(hidden)
        output = tf.squeeze(output, axis=-1)
        return output
