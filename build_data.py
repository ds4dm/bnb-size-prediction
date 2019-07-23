import re
import os
import numpy as np
import pickle
import multiprocessing as mp
import tensorflow as tf
import scip_utilities
from itertools import cycle
from pathlib import Path
from sampler import ActorSampler, Message


NB_TRAIN_SAMPLES = 100000
NB_TEST_SAMPLES = 20000
NB_SAMPLERS = 8


# Input files
def get_instance_id(path):
    return int(re.search(".*_(.*).lp$", str(path)).group(1))


def merge_folders(output_path):
    # Copy sample files
    sample_count = 0
    for samples in zip(*[sample_folder.glob("sample*") for sample_folder in output_path.iterdir()]):
        for sample in samples:
            sample.replace(output_path/f"sample_{sample_count}.pkl")
            sample_count += 1

    # Get benchmark
    benchmark = {}
    for sample_folder in output_path.iterdir():
        if sample_folder.is_dir():
            with (sample_folder/"benchmark.pkl").open("rb") as file:
                benchmark[sample_folder] = pickle.load(file)
    joint_benchmark = {}
    for results in benchmark.values():
        for instance in results:
            if instance not in joint_benchmark:
                joint_benchmark[instance] = {}
            for metric, value in results[instance].items():
                if metric not in joint_benchmark[instance]:
                    joint_benchmark[instance][metric] = value
                else:
                    joint_benchmark[instance][metric] += value
    with (output_path/"benchmark.pkl").open("wb") as file:
        pickle.dump(joint_benchmark, file)
    
    # Clean folder
    for sample_folder in output_path.iterdir():
        if sample_folder.is_dir():
            (sample_folder/"benchmark.pkl").unlink()
            sample_folder.rmdir()


input_path = Path("data/instances/setcover/")
train_instances = (input_path/"train_500r_1000c_0.05d").glob("*.lp")
train_instances = {path: get_instance_id(path) for path in train_instances}
train_instances = sorted(train_instances, key=train_instances.__getitem__)
test_instances = (input_path/"test_500r_1000c_0.05d").glob("*.lp")
test_instances = {path: get_instance_id(path) for path in test_instances}
test_instances = sorted(test_instances, key=test_instances.__getitem__)
parameters_path = "actor/pretrained-setcover/best_params.pkl"

# Train
# -----
actor_samplers = [ActorSampler(parameters_path, nb_solving_stats_samples=int(NB_TRAIN_SAMPLES/NB_SAMPLERS), 
                               id_=id_) for id_ in range(NB_SAMPLERS)]
for actor_sampler in actor_samplers:
    actor_sampler.start()

train_output_path = Path("data/bnb_size_prediction/setcover/train_500r_1000c_0.05d")
for count, instance_path in enumerate(train_instances):
    if count > NB_TRAIN_SAMPLES/(NB_SAMPLERS*10):
        break
    for actor_sampler in actor_samplers:
        actor_sampler.instance_queue.put({'type': Message.NEW_INSTANCE,
                                          'instance_path': str(instance_path),
                                          'solving_stats_output_dir': str(train_output_path)})

for actor_sampler in actor_samplers:
    actor_sampler.instance_queue.put({'type': Message.STOP})
merge_folders(train_output_path)

# Test
# ----
actor_samplers = [ActorSampler(parameters_path, nb_solving_stats_samples=int(NB_TEST_SAMPLES/NB_SAMPLERS), 
                               id_=id_) for id_ in range(NB_SAMPLERS)]
for actor_sampler in actor_samplers:
    actor_sampler.start()

test_output_path = Path("data/bnb_size_prediction/setcover/test_500r_1000c_0.05d")
for count, instance_path in enumerate(test_instances):
    if count > NB_TEST_SAMPLES/(NB_SAMPLERS*10):
        break
    for actor_sampler in actor_samplers:
        actor_sampler.instance_queue.put({'type': Message.NEW_INSTANCE,
                                          'instance_path': str(instance_path),
                                          'solving_stats_output_dir': str(test_output_path)})

for actor_sampler in actor_samplers:
    actor_sampler.instance_queue.put({'type': Message.STOP})
merge_folders(test_output_path)

    