import os
import queue
import time
import logging
import numpy as np
from pathlib import Path
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import scip_utilities
import utilities
import argparse
import pickle
from datetime import datetime

from agent import AsyncAgent as Agent
from actor.model import GCNPolicy as Actor


# Hyperparameters
SEED = 0
GAMMA = 1.0 # Do not change this!
REWARD_TYPE = 'nbnodes'
NB_SAMPLES = {'train':100000, 'valid':20000}


def tf_setup(gpu, seed):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    tf.enable_eager_execution(tfconfig)
    tf.set_random_seed(seed)


def calculate_return(rewards, gamma):
    ret = 0.0
    for i,r in enumerate(rewards):
        ret += r*(gamma**i)
    return ret


def update_benchmark(benchmark, result):
    instance = result['instance']
    if instance not in benchmark:
            benchmark[instance] = {'nb_nodes': [], 'nb_lp_iterations': []}
    benchmark[instance]['nb_nodes'].append(result['nb_nodes'])
    benchmark[instance]['nb_lp_iterations'].append(result['nb_lp_iterations'])
    with (output_path/"benchmark.pkl").open("wb") as file:
        pickle.dump(benchmark, file)
    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='Setcover or cauctions',
        type=str,
        choices=['setcover', 'cauctions'],
    )
    parser.add_argument(
        'split',
        help='Training or validation',
        type=str,
        choices=['train', 'valid'],
    )
    parser.add_argument(
        '-a', '--nb_agents',
        help='Number of parallel agents',
        type=int,
        default=8,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    files = {}
    if args.problem == 'setcover':
        data_folder = Path('data/instances/setcover')
        files['train'] = (data_folder/"train_500r_1000c_0.05d").glob("*.lp")
        files['valid'] = (data_folder/"valid_500r_1000c_0.05d").glob("*.lp")
    elif args.problem == 'cauctions':
        data_folder = Path('data/instances/cauctions')
        files['train'] = (data_folder/"train_100_500").glob("*.lp")
        files['valid'] = (data_folder/"valid_100_500").glob("*.lp")
    else:
        assert 0

    parameters_path = f"actor/{args.problem}/params.pkl"
    output_path = Path("data/samples_lara/bnb_size_prediction")/args.problem/args.split
    (output_path).mkdir(parents=True, exist_ok=True)

    # Logging
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')
    logging_header = (
    f"rl2branch log\n"
    f"-------------\n"
    f"Training started on {current_date} at {current_time}\n"
    )
    utilities.configure_logging(output_file="run.log", header=logging_header)
    logger = logging.getLogger("rl2branch")

    ############################################################

    # Instances
    instance_queue = queue.Queue()
    results_queue = queue.Queue()
    task_list = [(instance, seed) for instance in files[args.split] for seed in range(5)]
    for task in task_list:
        instance_queue.put(task)

    # Agents
    agents = [Agent(inQueue=instance_queue,
                    outQueue=results_queue,
                    reward_type=REWARD_TYPE,
                    logger = logger,
                    greedy=False,
                    record_tseries=True) for j in range(args.nb_agents)]
    tf_setup(args.gpu, SEED)
    actor = Actor()
    actor.restore_state(parameters_path)
    for agent in agents:
        agent.pass_actor_ref(actor)
        agent.start()

    # Sample
    nb_samples = NB_SAMPLES[args.split]
    benchmark = {}
    try:
        sample_count = 0
        while sample_count < nb_samples:
            live_agents = [agent for agent in agents if agent.finished == False]
            if len(live_agents) == 0: break
            print("Sample: ",sample_count)
            result = results_queue.get(block=True)
            critic_state = result['time_series']['solving_stats']
            nb_subsamples = np.ceil(0.05 * len(critic_state)).astype(int)
            subsample_ends = np.random.choice(np.arange(1, len(critic_state)+1), nb_subsamples, replace=False).tolist()
            for subsample_end in subsample_ends:
                subsample_states = scip_utilities.pack_solving_stats(critic_state[:subsample_end])
                nb_nodes_left = result['nb_nodes'] - result['time_series']['nb_nodes'][subsample_end-1]
                nb_lp_iterations_left = result['nb_lp_iterations'] - result['time_series']['nb_lp_iterations'][subsample_end-1]
                if sample_count < nb_samples:
                    sample_count += 1
                    sample_path = output_path/f"sample_{sample_count-1}.pkl"
                    with sample_path.open('wb') as f:
                        pickle.dump({'critic_state': subsample_states,
                                     'nb_nodes_left': nb_nodes_left,
                                     'nb_lp_iterations_left': nb_lp_iterations_left,
                                     'instance_path': str(result['instance']) }, f)
            if nb_subsamples > 0:
                benchmark = update_benchmark(benchmark,{'instance':str(result['instance']),
                                                    'nb_nodes': result['nb_nodes'],
                                                    'nb_lp_iterations':result['nb_lp_iterations']})

    finally:
        for agent in agents:
            agent.finished = True
            agent.stop()
        time.sleep(3)
        print("All agents were killed")
