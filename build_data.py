import re
import os
import numpy as np
import pickle
import multiprocessing as mp
import tensorflow as tf
import scip_utilities
from itertools import cycle, zip_longest
from pathlib import Path
from sampler import ActorSampler, Message


NB_TRAIN_SAMPLES = 100000
NB_VALID_SAMPLES = 20000
NB_SAMPLERS = 8

# Input files
def get_instance_id(path):
    return int(re.search(".*_(.*).lp$", str(path)).group(1))


def merge_folders(output_path):
    # Copy sample files
    sample_count = 0
    for samples in zip_longest(*[sample_folder.glob("sample*") for sample_folder in output_path.iterdir()]):
        for sample in samples:
            if sample is not None:
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

problem = "cauctions"
train_folder = Path(problem)/"train_100_500"
valid_folder = Path(problem)/"valid_100_500"
instance_path = Path("data/instances/")
parameters_path = Path("actor")/problem/"params.pkl"

train_instances = {path: get_instance_id(path) for path in (instance_path/train_folder).glob("*.lp")}
train_instances = sorted(train_instances, key=train_instances.__getitem__)
valid_instances = {path: get_instance_id(path) for path in (instance_path/valid_folder).glob("*.lp")}
valid_instances = sorted(valid_instances, key=valid_instances.__getitem__)

# Train
# -----
actor_samplers = [ActorSampler(parameters_path, nb_solving_stats_samples=int(NB_TRAIN_SAMPLES/NB_SAMPLERS), 
                               id_=id_) for id_ in range(NB_SAMPLERS)]
for actor_sampler in actor_samplers:
    actor_sampler.start()

train_output_path = Path("data/newscip_bnb_size_prediction")/train_folder
for count, instance_path in enumerate(train_instances):
    if count > NB_TRAIN_SAMPLES/(NB_SAMPLERS*10):
        break
    for actor_sampler in actor_samplers:
        actor_sampler.instance_queue.put({'type': Message.NEW_INSTANCE,
                                          'instance_path': str(instance_path),
                                          'solving_stats_output_dir': str(train_output_path)})

for actor_sampler in actor_samplers:
    actor_sampler.instance_queue.put({'type': Message.STOP})
for actor_sampler in actor_samplers:
    actor_sampler.join()

print("Merging train folders")
merge_folders(train_output_path)

# Valid
# -----
actor_samplers = [ActorSampler(parameters_path, nb_solving_stats_samples=int(NB_VALID_SAMPLES/NB_SAMPLERS), 
                               id_=id_) for id_ in range(NB_SAMPLERS)]
for actor_sampler in actor_samplers:
    actor_sampler.start()

valid_output_path = Path("data/newscip_bnb_size_prediction")/valid_folder
for count, instance_path in enumerate(valid_instances):
    if count > NB_VALID_SAMPLES/(NB_SAMPLERS*10):
        break
    for actor_sampler in actor_samplers:
        actor_sampler.instance_queue.put({'type': Message.NEW_INSTANCE,
                                          'instance_path': str(instance_path),
                                          'solving_stats_output_dir': str(valid_output_path)})

for actor_sampler in actor_samplers:
    actor_sampler.instance_queue.put({'type': Message.STOP})
for actor_sampler in actor_samplers:
    actor_sampler.join()

print("Merging valid folders")
merge_folders(valid_output_path)

    