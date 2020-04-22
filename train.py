import wandb
import os
import gzip
import argparse
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.contrib.eager as tfe
from pathlib import Path
from model import Model

config = {'lr_start': 1e-5,
          'lr_end': 1e-5,
          'max_epochs': 50}


def load_instance(filename):
    with gzip.open(filename, 'rb') as file:
        sample = pickle.load(file)
    features = tf.convert_to_tensor(sample['solving_stats'], dtype=tf.float32)
    response = tf.convert_to_tensor(sample['nb_nodes_left'], dtype=tf.float32)
    instance = sample['instance_path']
    return features, response, instance


def load_batch(batch_filenames):
    batch_features, batch_responses, batch_instances = [], [], []
    for count, filename in enumerate(batch_filenames):
        features, response, instance = load_instance(filename)
        batch_features.append(features)
        batch_responses.append(response)
        batch_instances.append(instance)
    batch_features = tf.stack(batch_features, axis=0)
    batch_responses = tf.stack(batch_responses, axis=0)
    batch_instances = tf.stack(batch_instances, axis=0)

    return batch_features, batch_responses, batch_instances


def get_feature_stats(data, folder):
    outfile = folder/"feature_stats.pickle"
    if outfile.exists():
        with outfile.open('rb') as file:
            feature_stats = pickle.load(file)
        feature_means = feature_stats['feature_means']
        feature_stds  = feature_stats['feature_stds']
    else:
        feature_means, feature_stds = [], []
        for features, _, _ in data:
            feature_means.append(tf.reduce_mean(features, axis=(0, 1)))
            mean = tf.expand_dims(tf.expand_dims(feature_means[-1], axis=0), axis=0)
            std  = tf.reduce_mean(tf.reduce_mean((features - mean) ** 2, axis=0), axis=0)
            feature_stds.append(tf.sqrt(std))
        feature_means = tf.reduce_mean(tf.stack(feature_means, axis=0), axis=0).numpy()
        feature_stds  = tf.reduce_mean(tf.stack(feature_stds, axis=0), axis=0).numpy()
        feature_stds[feature_stds < 1e-5] = 1.
        with outfile.open('wb') as file:
            pickle.dump({'feature_means': feature_means, 'feature_stds': feature_stds}, file)

    return feature_means, feature_stds


def learning_rate(episode):
    return (config['lr_start']-config['lr_end']) / np.e ** episode + config['lr_end']

def get_response_normalization(instances,benchmark):
    shift, scale = [], []
    for instance in instances:
        instance = instance.numpy().decode('utf-8')
        shift.append(0.0)
        scale.append(np.mean(benchmark[instance]['nb_nodes']))
        #shift.append(np.mean(benchmark[instance]['nb_nodes']))
        #scale.append(np.mean(benchmark[instance]['nb_nodes'])/np.sqrt(12) + np.std(benchmark[instance]['nb_nodes']))
    shift = tf.cast(tf.stack(shift, axis=0), tf.float32)
    scale = tf.cast(tf.stack(scale, axis=0), tf.float32)
    return shift, scale

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='Setcover or cauctions',
        type=str,
        choices=['setcover', 'cauctions'],
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
    rng = np.random.RandomState(0)
    wandb.init(project="bnb-size-prediction", config=config)

    data_folder = Path('data/bnb_size_prediction')/args.problem
    train_folder = data_folder/"train"
    valid_folder  = data_folder/"valid"
    output_folder = Path('results')/args.problem
    output_folder.mkdir(parents=True, exist_ok=True)

    train_filenames = [str(filename) for filename in train_folder.glob('sample*.pkl')]
    train_data = tf.data.Dataset.from_tensor_slices(train_filenames).batch(32)
    train_data = train_data.map(lambda x: tf.py_func(load_batch, [x], [tf.float32, tf.float32, tf.string]))
    train_data = train_data.prefetch(1)
    with (train_folder/"benchmark.pkl").open("rb") as file:
        train_benchmark = pickle.load(file)

    valid_filenames = [str(filename) for filename in valid_folder.glob('sample*.pkl')]
    valid_data = tf.data.Dataset.from_tensor_slices(valid_filenames).batch(128)
    valid_data = valid_data.map(lambda x: tf.py_func(load_batch, [x], [tf.float32, tf.float32, tf.string]))
    valid_data = valid_data.prefetch(1)
    with (valid_folder/"benchmark.pkl").open("rb") as file:
        valid_benchmark = pickle.load(file)

    feature_means, feature_stds = get_feature_stats(train_data, train_folder)

    model = Model(feature_means, feature_stds)
    optimizer = tf.train.AdamOptimizer(lambda: lr)

    best_valid_loss = np.inf
    for epoch in range(config['max_epochs']):
        K.backend.set_learning_phase(1) # Set train

        epoch_train_filenames = rng.choice(train_filenames, len(train_filenames), replace=False)
        train_data = tf.data.Dataset.from_tensor_slices(epoch_train_filenames).batch(32)
        train_data = train_data.map(lambda x: tf.py_func(load_batch, [x], [tf.float32, tf.float32, tf.string]))
        train_data = train_data.prefetch(1)

        train_loss = []
        rel_loss = []
        for count, (features, unnorm_responses, instances) in enumerate(train_data):
            shift, scale = get_response_normalization(instances,train_benchmark)
            responses = (unnorm_responses - shift) / scale

            lr = learning_rate(epoch)
            with tf.GradientTape() as tape:
                predictions = model(features)
                loss = tf.reduce_mean(tf.square(predictions - responses))
            grads = tape.gradient(target=loss, sources=model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))
            train_loss.append(loss)
            rel_loss.append(tf.reduce_mean(scale * tf.abs(predictions - responses) / (unnorm_responses + 1)))
            if count % 500 == 0:
                print(f"Epoch {epoch}, batch {count}, loss {loss:.4f}")
        train_loss = tf.reduce_mean(train_loss)
        rel_loss = tf.reduce_mean(rel_loss)
        wandb.log({'train_loss': train_loss.numpy(),
                   'train_transformed_loss': rel_loss.numpy()}, step=epoch)
        print(f"Epoch {epoch}, train loss {train_loss:.4f}")

        K.backend.set_learning_phase(0) # Set valid
        valid_loss = []
        rel_loss = []
        for batch_count, (features, unnorm_responses, instances) in enumerate(valid_data):
            shift, scale = get_response_normalization(instances,valid_benchmark)
            responses = (unnorm_responses - shift) / scale

            predictions = model(features)
            loss = tf.reduce_mean(tf.square(predictions - responses))
            valid_loss.append(loss)
            rel_loss.append(tf.reduce_mean(scale * tf.abs(predictions - responses) / (unnorm_responses + 1)))
        valid_loss = tf.reduce_mean(valid_loss)
        rel_loss = tf.reduce_mean(rel_loss)
        wandb.log({'valid_loss': valid_loss.numpy(),
                   'valid_transformed_loss': rel_loss.numpy()}, step=epoch)
        print(f"Epoch {epoch}, validation loss {valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print(" * New best validation loss *")
            model.save_state(output_folder/"best_params.pkl")
