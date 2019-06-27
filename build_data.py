import os
import numpy as np
import pickle
import multiprocessing as mp
import tensorflow as tf
import scip_utilities
from itertools import cycle
from pathlib import Path
from sampler import ActorSampler, Message


NB_TRAINING_SAMPLES = 100000
NB_TEST_SAMPLES = 20000
NB_SAMPLERS = 8

# Input files
input_path = Path("data/instances/setcover/")
train_files = (input_path/"train_500r_1000c_0.05d").glob("*.lp")
test_files = (input_path/"test_500r_1000c_0.05d").glob("*.lp")
parameters_path = "actor/pretrained-setcover/best_params.pkl"

# Output files
output_path = Path("data/bnb_size_prediction/baseline/test")
actor_samplers = [ActorSampler(parameters_path, nb_solving_stats_samples=NB_TRAINING_SAMPLES, id_=0) for id_ in range(NB_SAMPLERS)]

for actor_sampler in actor_samplers:
    actor_sampler.start()

for count, instance_path in train_files:
    if count > 100:
        for actor_sampler in actor_samplers:
            actor_sampler.instance_queue.put({'type': Message.NEW_INSTANCE,
                                              'instance_path': str(instance_path),
                                              'solving_stats_output_dir': str(output_path/"train_500r_1000c_0.05d")})

for actor_sampler in actor_samplers:
    actor_sampler.instance_queue.put({'type': Message.STOP})
