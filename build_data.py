import os
import queue
import numpy as np
from pathlib import Path
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import scip_utilities
import pickle

from agent import AsyncAgent as Agent
from actor.model import GCNPolicy as Actor

def tf_setup(gpu,seed):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    tf.enable_eager_execution(tfconfig)
    tf.set_random_seed(seed)

def calculate_return(rewards,gamma):
    ret = 0.0
    for i,r in enumerate(rewards):
        ret += r*(gamma**i)
    return ret

def update_benchmark(benchmark,result):
    instance = result['instance']
    if instance not in benchmark:
            benchmark[instance] = {'nb_nodes_final': [], 'nb_lp_iterations_final': []}
    benchmark[instance]['nb_nodes_final'].append(result['nb_nodes_final'])
    benchmark[instance]['nb_lp_iterations_final'].append(result['nb_lp_iterations_final'])
    with (output_path/"benchmark.pkl").open("wb") as file:
        pickle.dump(benchmark, file)
    return benchmark

# Parameters
N_AGENTS = 8
problem = 'cauctions'
type = 'test'
gpu = 1
seed = 0
gamma = 0.99
reward = 'gap'
nb_samples = {'train':100000,'test':20000}

files = {}
if problem == 'setcover':
    data_folder = Path('data/instances/setcover')
    files['train'] = (data_folder/"train_500r_1000c_0.05d").glob("*.lp")
    files['test'] = (data_folder/"test_500r_1000c_0.05d").glob("*.lp")
elif problem == 'cauctions':
    data_folder = Path('data/instances/cauctions')
    files['train'] = (data_folder/"train_100_500").glob("*.lp")
    files['test'] = (data_folder/"test_100_500").glob("*.lp")
else:
    assert 0

parameters_path = f"actor/{problem}/params.pkl"
output_path = Path(f"data/bnb_size_prediction/"+type)
(output_path).mkdir(parents=True, exist_ok=True)


############################################################

# Instances
instance_queue = queue.Queue()
results_queue = queue.Queue()
task_list = [(instance,seed) for instance in files[type] for seed in range(5)]
for task in task_list:
    instance_queue.put(task)

# Agents
agents = [Agent(policy=str(j),
                inQueue=instance_queue,
                outQueue=results_queue,
                reward_type=reward,
                greedy=False,
                record_states=True) for j in range(N_AGENTS)]
tf_setup(gpu,seed)
actor = Actor()
actor.restore_state(parameters_path)
for agent in agents:
    agent.pass_actor_ref(actor)
    agent.start()

# Sample
nb_samples = nb_samples[type]
benchmark = {}
try:
    sample_count = 0
    while sample_count < nb_samples:
        print("Sample: ",sample_count)
        result = results_queue.get(block=True)
        if result is None:
            break
        c_states = result['c_states']
        nb_subsamples = np.ceil(0.05 * len(c_states)).astype(int)
        subsample_ends = np.random.choice(np.arange(1, len(c_states)+1), nb_subsamples, replace=False).tolist()
        for subsample_end in subsample_ends:
            subsample_stats = scip_utilities.pack_solving_stats(c_states[:subsample_end])
            return_left = calculate_return(result['rewards'][subsample_end-1:],gamma)
            nb_nodes_left = result['nb_nodes_final'] - result['nb_nodes'][subsample_end-1]
            nb_lp_iterations_left = result['nb_lp_iterations_final'] - result['nb_lp_iterations'][subsample_end-1]
            if sample_count < nb_samples:
                sample_count += 1
                sample_path = output_path/f"sample_{sample_count-1}.pkl"
                with sample_path.open('wb') as f:
                    pickle.dump({'c_states': subsample_stats,
                                 'nb_nodes_left': nb_nodes_left,
                                 'nb_lp_iterations_left': nb_lp_iterations_left,
                                 'return_left': return_left,
                                 'instance_path': str(result['instance']) }, f)
        if nb_subsamples > 0:
            benchmark = update_benchmark(benchmark,{'instance':str(result['instance']),
                                                'nb_nodes_final': result['nb_nodes_final'],
                                                'nb_lp_iterations_final':result['nb_lp_iterations_final']})

finally:
    for agent in agents: agent.kill()
    print("All agents were killed")
