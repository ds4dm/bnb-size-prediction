import pickle
import tensorflow as tf
import tensorflow.keras as K


SEQUENCE_LENGTH = 50
FEATURE_SIZE = 36
NODE_FEATURE_SIZE = 6
DEPTH = 16


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

class MultiHeadAttentionLayer(K.layers.Layer):
    def __init__(self, num_heads, depth, d_q, d_v=None):
        super(MultiHeadAttentionLayer, self).__init__()

        self.num_heads = num_heads
        self.depth = depth
        self.units = depth * num_heads

        self.q_layer = K.layers.Dense(self.units)
        self.k_layer = K.layers.Dense(self.units)
        self.v_layer = K.layers.Dense(self.units)
        self.linear_layer = K.layers.Dense(depth, activation='relu')

        # Build right away
        if d_v == None: d_v = d_q
        self.build((d_q,d_v))

    def build(self,input_shapes):
        d_q, d_v = input_shapes
        if not self.built:
            self.q_layer.build(d_q)
            self.k_layer.build(d_q)
            self.v_layer.build(d_v)
            self.linear_layer.build(self.units)
            self.built = True

    def split(self, x):
        # Split last dimension into heads
        shape = list(x.shape)[:-1] + [self.num_heads,self.depth]
        x = tf.reshape(x, tuple(shape))
        x = self.transpose(x)
        return x

    def transpose(self,x):
        # Transpose dimensions -2 and -3
        shape = list(x.shape)
        perm = [*range(len(shape))]
        perm[-2] = len(shape)-3
        perm[-3] = len(shape)-2
        return tf.transpose(x, perm=perm)

    def get_attention_mask(self,mask):
        """ Mask must be a binary tensor of dimensions
            [batch_size, sequence_length, node_buffer_size]"""
        n = mask.shape[-1]
        rep = [1 for dim in range(len(mask.shape))] + [n]
        mask = tf.expand_dims(mask,axis=-1)
        mask = tf.tile(mask,rep)

        # Create transposed tensor
        shape = list(mask.shape)
        perm = [*range(len(shape))]
        perm[-1] = len(shape)-2
        perm[-2] = len(shape)-1
        mask_t = tf.transpose(mask,perm=perm)
        mask = tf.multiply(mask,mask_t)

        # Repeat for all attention heads
        mask = tf.expand_dims(mask,axis=-3)
        rep = [1 for dim in range(len(mask.shape))]
        rep[-3] = self.num_heads
        mask = tf.tile(mask,rep)
        return mask

    def get_output_mask(self,mask):
        rep = [1 for dim in range(len(mask.shape))] + [self.depth]
        mask = tf.expand_dims(mask,axis=-1)
        mask = tf.tile(mask,rep)
        return mask

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        qk = tf.matmul(Q, K, transpose_b=True)
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention = qk / tf.math.sqrt(dk)

        if mask is not None:
            mask = self.get_attention_mask(mask)
            scaled_attention += (mask * -1e9)

        weights = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(weights, V)

        return output, weights

    def call(self, Q, K=None, V=None, mask=None):
        if K == None and V==None:
            # Self attention
            K = Q
            V = Q

        # Run through linear layers
        Q = self.q_layer(Q)
        K = self.k_layer(K)
        V = self.v_layer(V)

        # Split the heads
        Q = self.split(Q)
        K = self.split(K)
        V = self.split(V)

        # Run through attention
        attention_output, weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Prepare for the rest of processing
        attention_output= self.transpose(attention_output)
        shape = list(attention_output.shape)[:-2] + [self.units]
        concat_attention = tf.reshape(attention_output, tuple(shape))

        # Run through final linear layer
        output = self.linear_layer(concat_attention)
        if mask is not None:
            output = tf.multiply(output,self.get_output_mask(mask))

        return output #, weights

class Model(BaseModel):
    def __init__(self, feature_means=None, feature_stds=None):
        super().__init__()
        if feature_means is None:
            feature_means = tf.zeros(FEATURE_SIZE)
        if feature_stds is None:
            feature_stds = tf.ones(FEATURE_SIZE)
        self.norm = K.layers.Dense(feature_means.shape[0], trainable=False,
            kernel_initializer=lambda shape, dtype, partition_info: tf.diag(1/feature_stds),
            bias_initializer=lambda shape, dtype, partition_info: -feature_means/feature_stds)

        self.self_attention = MultiHeadAttentionLayer(num_heads=2, depth=DEPTH, d_q=NODE_FEATURE_SIZE)

        self.conv = K.Sequential([
            K.layers.Conv1D(filters=64, kernel_size=5, padding="same", activation='relu',
                            kernel_initializer=K.initializers.Orthogonal(),
                            input_shape=(SEQUENCE_LENGTH, FEATURE_SIZE+DEPTH)),
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
        self.build([
            (None, SEQUENCE_LENGTH, FEATURE_SIZE),
            (None, SEQUENCE_LENGTH, None, NODE_FEATURE_SIZE),
            (None, SEQUENCE_LENGTH, None)])

        # save / restore fix
        self.variables_topological_order = [v.name for v in self.variables]

    def build(self, input_shapes):
        solving_stats_shape , open_node_stats_shape, mask_shape = input_shapes
        _, sequence_length, feature_size = solving_stats_shape
        _, sequence_length, _, node_feature_size = open_node_stats_shape

        if not self.built:
            self.norm.build(feature_size)
            self.self_attention.build((node_feature_size,node_feature_size))
            self.conv.build((None, sequence_length, feature_size + DEPTH))
            self.dense.build((None, 448))
            self.built = True

    def call(self, inputs):
        solving_stats, open_node_stats, mask = inputs

        # Process open node stats
        open_node_stats = self.self_attention(open_node_stats,mask=mask)
        open_node_stats = tf.math.reduce_max(open_node_stats,axis=-2)

        # Stack and process
        solving_stats = self.norm(solving_stats)
        solving_stats = tf.concat([solving_stats,open_node_stats],axis=-1)
        hidden = self.conv(solving_stats)
        output = self.dense(hidden)
        output = tf.squeeze(output, axis=-1)
        return output
