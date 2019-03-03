import os
import numpy as np
import pickle
import multiprocessing as mp
import tensorflow as tf
import scip_utilities
from pathlib import Path
from config import *
from sampler import ActorSampler, Message


NB_TRAINING_SAMPLES = 100000
NB_TEST_SAMPLES = 20000
NB_SAMPLERS = 8


def generate_samples(instance_queue, results_queue, nb_samples, 
                     files, output_path, instance_batch_size=NB_SAMPLERS):
    sample_count = 0
    files = iter(files)

    print(f"Start of sample generation in in {output_path}")
    while sample_count < nb_samples:
        # Send orders
        for _ in range(instance_batch_size):
            instance_queue.put({'type': Message.NEW_INSTANCE,
                                'instance_path': str(next(files))})

        # Receive results
        results = []
        for _ in range(instance_batch_size):
            results.append(results_queue.get())

        # Process results
        for result in results:
            solving_stats, nb_nodes_total = result['solving_stats'], result['nb_nodes_total']
            nb_subsamples = np.ceil(0.05 * len(solving_stats)).astype(int)
            subsample_ends = np.random.choice(np.arange(1, len(solving_stats)), nb_subsamples, replace=False).tolist()
            for subsample_end in subsample_ends:
                subsample_stats = solving_stats[max(subsample_end-SEQUENCE_LENGTH, 0):subsample_end]
                subsample_stats = {name: np.asarray([s[name] for s in subsample_stats]) 
                                   for name in subsample_stats[0].keys()}
                subsample_stats = scip_utilities.normalize_solving_stats(subsample_stats, length=50)
                subsample_stats = np.stack([subsample_stats[feature_name] 
                                         for feature_name in SOLVING_STATS_FEATURES], axis=-1)
                nb_nodes_left = nb_nodes_total - (subsample_end + 1)

            if sample_count < nb_samples:
                sample_count += 1
                sample_path = output_path/f"sample_{sample_count-1}.pkl"
                if sample_count % 100 == 1:
                    print(f"Saving {sample_path}")
                with sample_path.open('wb') as f:
                    pickle.dump((subsample_stats, nb_nodes_left), f)
    print("Done!")


if __name__ == "__main__":
    # Input files
    data_folder = Path("data/instances/setcover/")
    train_files = (data_folder/"train_500r_1000c_0.05d").glob("*.lp")
    test_files = (data_folder/"test_500r_1000c_0.05d").glob("*.lp")
    parameters_path = "actor/pretrained-setcover/best_params.pkl"

    # Output files
    output_path = Path("data/bnb_node_prediction/setcover")
    (output_path/"train_500r_1000c_0.05d").mkdir(parents=True, exist_ok=True)
    (output_path/"test_500r_1000c_0.05d").mkdir(parents=True, exist_ok=True)

    
    instance_queue, results_queue = mp.Queue(), mp.Queue()
    actor_samplers = [ActorSampler(parameters_path, instance_queue, results_queue, seed=i) 
                      for i in range(NB_SAMPLERS)]

    for actor_sampler in actor_samplers:
        actor_sampler.start()
    generate_samples(instance_queue, results_queue, 
                     NB_TRAINING_SAMPLES, train_files, 
                     output_path=output_path/"train_500r_1000c_0.05d")
    generate_samples(instance_queue, results_queue, 
                     NB_TEST_SAMPLES, test_files, 
                     output_path=output_path/"test_500r_1000c_0.05d")
    for _ in actor_samplers:
        instance_queue.put({'type': Message.STOP})
