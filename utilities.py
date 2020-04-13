"""
Miscellaneous utilities.
"""
import os
import json
import pickle
import logging
import operator
import functools
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from pathlib import Path
from collections import deque


# Multiprocessing decorators
def internal(method):
    @functools.wraps(method)
    def protected_method(self, *args, **kwargs):
        assert mp.current_process() == self
        return method(self, *args, **kwargs)
    return protected_method


def external(method):
    @functools.wraps(method)
    def protected_method(self, *args, **kwargs):
        assert mp.current_process() != self
        return method(self, *args, **kwargs)
    return protected_method


def valid_seed(seed):
    """Check whether seed is a valid random seed or not."""
    seed = int(seed)
    if seed < 0 or seed > 2**32 - 1:
        raise argparse.ArgumentTypeError(
                "seed must be any integer between 0 and 2**32 - 1 inclusive")
    return seed

class FormatterWithHeader(logging.Formatter):
    """
    From
    https://stackoverflow.com/questions/33468174/write-header-to-a-python-log-file-but-only-if-a-record-gets-written
    """
    def __init__(self, header, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
        self.header = header
        self.format = self.first_line_format

    def first_line_format(self, record):
        self.format = super().format
        return self.header + "\n" + self.format(record)


def configure_logging(output_file, header=""):
    logger = logging.getLogger("rl2branch")
    logger.setLevel(logging.DEBUG)

    formatter = FormatterWithHeader(header=header,
                                    fmt='[%(asctime)s %(levelname)-8s]  %(threadName)-12s  %(message)s',
                                    datefmt='%H:%M:%S')

    os.makedirs("logs/", exist_ok=True)
    file_handler = logging.FileHandler("logs/" + output_file, 'w', 'utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def load_config(filename):
    with open(filename, 'r') as file:
        config = json.load(file)
    return config


def batch_actor_states(constraint_features, edge_indices, edge_features, variable_features):
    nb_constraints = [features.shape[0] for features in constraint_features]
    nb_variables = [features.shape[0] for features in variable_features]

    constraint_features = tf.concat(constraint_features, axis=0)
    index_shifts = tf.cumsum([[0] + nb_constraints[:-1], [0] + nb_variables[:-1]], axis=1)
    edge_indices = tf.concat([indices + index_shifts[:, i:(i+1)]
                              for i, indices in enumerate(edge_indices)], axis=1)
    edge_features = tf.concat(edge_features, axis=0)
    variable_features = tf.concat(variable_features, axis=0)
    nb_constraints = tf.stack(nb_constraints, axis=0)
    nb_variables = tf.stack(nb_variables, axis=0)

    state = (constraint_features, edge_indices, edge_features, variable_features, nb_constraints, nb_variables)
    return state


def StrongBranchingSamples(seed=0):
    random = np.random.RandomState(seed)
    train_files = [str(x) for x in Path('data/samples/setcover/500r_1000c_0.05d/train').glob('sample_*.pkl')]
    while True:
        epoch_train_files = random.choice(train_files, 312 * 32, replace=True)
        train_data = tf.data.Dataset.from_tensor_slices(epoch_train_files)
        train_data = train_data.batch(32).map(lambda x: tf.py_func(
            load_batch, [x], [tf.float32, tf.int32, tf.float32, tf.float32, tf.int32, tf.int32, tf.int32]))
        train_data = train_data.prefetch(1)

        for batch in train_data:
            constraint_features, edge_indices, edge_features, variable_features, \
                nb_constraints, nb_variables, actions = batch
            batched_states = (constraint_features, edge_indices, edge_features, variable_features, nb_constraints, nb_variables)
            yield batched_states, actions


def load_batch(sample_files):
    """
    Loads and concatenates a bunch of samples into one mini-batch.
    """
    constraint_features = []
    edge_indices = []
    edge_features = []
    variable_features = []
    actions = []

    # load samples
    for filename in sample_files:
        with open(filename, 'rb') as f:
            sample = pickle.load(f)

        state, action, *_ = sample['data']
        constraint, edge, variable = state
        constraint_features.append(constraint['values'])
        edge_indices.append(edge['indices'])
        edge_features.append(edge['values'])
        variable_features.append(variable['values'])
        actions.append(action)

    nb_constraints = [constraint.shape[0] for constraint in constraint_features]
    nb_variables = [variable.shape[0] for variable in variable_features]

    # concatenate samples in one big graph
    constraint_features = np.concatenate(constraint_features, axis=0)
    # edge indices have to be adjusted accordingly
    shift = np.cumsum([[0] + nb_constraints[:-1],
                       [0] + nb_variables[:-1]   ], axis=1)
    edge_indices = np.concatenate([edge_index + shift[:, edge_count:(edge_count+1)]
                                   for edge_count, edge_index in enumerate(edge_indices)], axis=1)
    variable_features = np.concatenate(variable_features, axis=0)
    edge_features = np.concatenate(edge_features, axis=0)
    actions = np.array(actions)

    # convert to tensors
    constraint_features = tf.convert_to_tensor(constraint_features, dtype=tf.float32)
    edge_indices = tf.convert_to_tensor(edge_indices, dtype=tf.int32)
    edge_features = tf.convert_to_tensor(edge_features, dtype=tf.float32)
    variable_features = tf.convert_to_tensor(variable_features, dtype=tf.float32)
    nb_constraints = tf.convert_to_tensor(nb_constraints, dtype=tf.int32)
    nb_variables = tf.convert_to_tensor(nb_variables, dtype=tf.int32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)

    return constraint_features, edge_indices, edge_features, variable_features, nb_constraints, nb_variables, actions

class SmoothedStatistic:
    def __init__(self, buffer_size, patience=np.inf):
        self.data = deque(maxlen=buffer_size)
        self.best = np.inf

        self._patience = patience
        self._patience_counter = np.inf

    def append(self, value):
        self.data.append(value)

    @property
    def value(self):
        return np.mean(self.data)

    def breakthrough(self, tolerance=1e-3):
        if len(self.data) == self.data.maxlen and self.value < self.best - tolerance:
            self.best = self.value
            self._patience_counter = self._patience
            return True
        else:
            self._patience_counter -= 1
            return False

    def patience_exhausted(self):
        if self._patience_counter <= 0:
            self._patience_counter = self._patience
            return True
        else:
            return False

class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.

        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))

        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences

        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix

        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)

class LinearSchedule:
    def __init__(self,timesteps,initial,final):
        # value = a*t + b
        self.a = (final - initial) / timesteps
        self.b = initial
        self.clip = final
        self.sign = 2*(self.a>=0)-1
        self.value = initial

    def set_value(self,t):
        self.value = self.sign*min(self.sign*(self.a*t + self.b) , self.sign*self.clip)
