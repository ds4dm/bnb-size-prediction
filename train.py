import os
import argparse
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.contrib.eager as tfe
from pathlib import Path
from .config import *
from .model import Model


lr_start = 1e-4
lr_end = 1e-5
max_epochs = 50


def normalize_solving_stats(solving_stats, length=50):
    solving_stats = {name: np.pad(vals[-length:], (max(length-len(vals), 0), 0), mode='edge') for name, vals in solving_stats.items()}
    
    nnodes_done = solving_stats['ninternalnodes'] + solving_stats['nfeasibleleaves'] + solving_stats['ninfeasibleleaves'] + solving_stats['nobjlimleaves']
    solving_stats['nnodes_done'] = nnodes_done
    lp_obj_norm = [(v - lb) / ((ub - lb) if ub > lb else 1) for v, lb, ub in zip(solving_stats['lp_obj'], solving_stats['dualbound'], solving_stats['primalbound'])]
    solving_stats['lp_obj_norm'] = lp_obj_norm
    lp_obj_normfirst = [(v - solving_stats['dualbound'][0]) / ((solving_stats['primalbound'][0] - solving_stats['dualbound'][0]) if solving_stats['primalbound'][0] > solving_stats['dualbound'][0] else 1) for v in solving_stats['lp_obj']]
    solving_stats['lp_obj_normfirst'] = lp_obj_normfirst
    avgdualbound_normfirst = [(v - solving_stats['dualbound'][0]) / ((solving_stats['primalbound'][0] - solving_stats['dualbound'][0]) if solving_stats['primalbound'][0] > solving_stats['dualbound'][0] else 1) for v in solving_stats['avgdualbound']]
    solving_stats['avgdualbound_normfirst'] = avgdualbound_normfirst
    avgpseudocostscore_norm = [(v - lb) / ((ub - lb) if ub > lb else 1) for v, lb, ub in zip(solving_stats['avgpseudocostscore'], solving_stats['dualbound'], solving_stats['primalbound'])]
    solving_stats['avgpseudocostscore_norm'] = avgpseudocostscore_norm
    avgpseudocostscore_normfirst = [(v - solving_stats['dualbound'][0]) / ((solving_stats['primalbound'][0] - solving_stats['dualbound'][0]) if solving_stats['primalbound'][0] > solving_stats['dualbound'][0] else 1) for v in solving_stats['avgpseudocostscore']]
    solving_stats['avgpseudocostscore_normfirst'] = avgpseudocostscore_normfirst              
    for k in (10, 25, 50, 75, 90):
        quint = f'opennodes_{k}quant'
        quint_norm = f'opennodes_{k}quant_norm'
        quint_normfirst = f'opennodes_{k}quant_normfirst'
        opennodes_quint_norm = [(v - lb) / ((ub - lb) if ub > lb else 1) for v, lb, ub in zip(solving_stats[quint], solving_stats['dualbound'], solving_stats['primalbound'])]
        solving_stats[quint_norm] = opennodes_quint_norm
        opennodes_quint_normfirst = [(v - solving_stats['dualbound'][0]) / ((solving_stats['primalbound'][0] - solving_stats['dualbound'][0]) if solving_stats['primalbound'][0] > solving_stats['dualbound'][0] else 1) for v in solving_stats[quint]]
        solving_stats[quint_normfirst] = opennodes_quint_normfirst
    return solving_stats


def load_instance(filename):
    with open(filename, 'rb') as file:
        solving_stats, nb_nodes, nb_nodes_total = pickle.load(file)

    solving_stats = normalize_solving_stats(solving_stats, length=SEQUENCE_LENGTH)
    features = np.stack([solving_stats[feature_name] for feature_name in FEATURE_NAMES], axis=-1)
    features = tf.convert_to_tensor(features, dtype=tf.float32)
    response = tf.convert_to_tensor(nb_nodes_total - nb_nodes, dtype=tf.float32)
    return features, response


def load_batch(batch_filenames):
    batch_features, batch_responses = [], []
    for count, filename in enumerate(batch_filenames):
        features, response = load_instance(filename)
        batch_features.append(features)
        batch_responses.append(response)
    batch_features = tf.stack(batch_features, axis=0)
    batch_responses = tf.stack(batch_responses, axis=0)

    return batch_features, batch_responses


def learning_rate(episode):
    return (lr_start-lr_end) / np.e ** episode + lr_end


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    args = parser.parse_args()
    
    if args.gpu == -1:
        print(f"Using CPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        print(f"Using GPU {args.gpu}")
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    tf.enable_eager_execution(tfconfig)
    tf.set_random_seed(seed=0)
    
    data_folder = Path('../../data/pickle/dataset_new_model')

    train_filenames = [str(filename) for filename in data_folder.glob('train*.pkl')]
    train_data = tf.data.Dataset.from_tensor_slices(train_filenames).batch(32)
    train_data = train_data.map(lambda x: tf.py_func(load_batch, [x], [tf.float32, tf.float32]))
    train_data = train_data.prefetch(1)

    test_filenames = [str(filename) for filename in data_folder.glob('test*.pkl')]
    test_data = tf.data.Dataset.from_tensor_slices(test_filenames).batch(128)
    test_data = test_data.map(lambda x: tf.py_func(load_batch, [x], [tf.float32, tf.float32]))
    test_data = test_data.prefetch(1)

    if (data_folder/"stats.pickle").exists():
        with (data_folder/"stats.pickle").open('rb') as file:
            feature_stats = pickle.load(file)
        feature_means = feature_stats['feature_means']
        feature_stds = feature_stats['feature_stds']
    else:
        feature_means, feature_stds = [], []
        for features, responses in train_data:
            feature_means.append(tf.reduce_mean(features, axis=(0, 1)))
            mean = tf.expand_dims(tf.expand_dims(feature_means[-1], axis=0), axis=0)
            std = tf.reduce_mean(tf.reduce_mean((features - mean) ** 2, axis=0), axis=0)
            feature_stds.append(tf.sqrt(std))
        feature_means = tf.reduce_mean(tf.stack(feature_means, axis=0), axis=0).numpy()
        feature_stds = tf.reduce_mean(tf.stack(feature_stds, axis=0), axis=0).numpy()
        feature_stds[feature_stds < 1e-5] = 1.

        with (data_folder/"stats.pickle").open('wb') as file:
            pickle.dump({'feature_means': feature_means, 'feature_stds': feature_stds}, file)

    model = Model(feature_means, feature_stds)
    optimizer = tf.train.AdamOptimizer(lambda: lr)

    for epoch in range(max_epochs):
        K.backend.set_learning_phase(1) # Set train
        train_loss = []
        for count, (features, responses) in enumerate(train_data):
            lr = learning_rate(epoch)
            with tf.GradientTape() as tape:
                predictions = model(features)
                loss = tf.reduce_mean(tf.square(tf.log(predictions + 1e-5) - tf.log(responses + 1e-5)))
            grads = tape.gradient(target=loss, sources=model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))
            train_loss.append(loss)
            if count % 500 == 0:
                print(f"Epoch {epoch}, batch {count}, loss {loss:.2f}")
        train_loss = tf.reduce_mean(train_loss)
        print(f"Epoch {epoch}, train loss {train_loss:.2f}")

        K.backend.set_learning_phase(0) # Set test
        test_loss = []
        for batch_count, (features, responses) in enumerate(test_data):
            predictions = model(features)
            loss = tf.reduce_mean(tf.square(tf.log(predictions + 1e-5) - tf.log(responses + 1e-5)))
            test_loss.append(loss)
        test_loss = tf.reduce_mean(test_loss)
        print(f"Epoch {epoch}, test loss {test_loss:.2f}")

    model.save_state("pretrained-setcover/best_params.pkl")
