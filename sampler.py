"""
Code for the actor sampler, for generating datasets for the critic.
"""
import os
import time
import enum
import gzip
import pickle
import logging
import traceback
import psutil
import numpy as np
import multiprocessing as mp
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import pyscipopt
import scip_utilities
# from wurlitzer import sys_pipes
from actor.model import GCNPolicy
from scipy.special import softmax
from pathlib import Path


class ActorSampler(mp.Process, pyscipopt.Branchrule):
    def __init__(self, parameters_path, nb_solving_stats_samples, id_):
        super().__init__()
        self.parameters_path = parameters_path
        self.instance_queue = mp.SimpleQueue()
        self.nb_solving_stats_samples = nb_solving_stats_samples
        self.id = id_
        self.seed = id_
        self.actor = None
        self._sample_count = 0
        self._benchmark = {}
        self._logger = None
        self._reward = None
        self._return = None
        self._reoptimization_count = None
        self._nb_steps = None
        self._actor_weights = None

    def run(self):
        self.configure_logger()
        try:
            self.load_actor()
            # DEBUG
            while True:
                message = self.instance_queue.get()
                if message['type'] == Message.NEW_INSTANCE:
                    instance_path = str(message['instance_path'])
                    solving_stats_output_dir = message['solving_stats_output_dir']
                    if solving_stats_output_dir is not None:
                        solving_stats_output_dir = Path(solving_stats_output_dir)/str(self.id)
                        solving_stats_output_dir.mkdir(parents=True, exist_ok=True)
                elif message['type'] == Message.STOP:
                    break
                else:
                    raise ValueError(f"Unrecognized message {message}")

                self.actor = tfe.defun(self._actor_weights.call,
                                       input_signature=self._actor_weights.input_signature)
                tf.set_random_seed(self.seed)
                tf.reset_default_graph()
                model = pyscipopt.Model()
                model.setIntParam('display/verblevel', 0)

                model.readProblem(instance_path)
                scip_utilities.init_scip_params(model, seed=self.seed)

                recorder = SolvingStatsRecorder(sampler=self)
                model.includeEventhdlr(recorder, "SolvingStatsRecorder", "")
                model.includeBranchrule(branchrule=self,
                    name="My branching rule", desc="",
                    priority=666666, maxdepth=-1, maxbounddist=1)
                self._reward = NbNodesRewards(model)
                self._return = 0.0
                self._nb_steps = 0

                # DEBUG
                self._logger.info(f"Solving {instance_path}")
                print(f"{self.name}: solving {instance_path}")
                model.optimize()
                if self._nb_steps > 0:
                    self._return += self._reward()
                    self.save_results(model, recorder, instance_path, solving_stats_output_dir)
                self._logger.info(f"Done solving {instance_path}")
                model.freeProb()
            self._logger.info(f"Done!")
        except Exception as exception:
            info = type(exception), exception, exception.__traceback__
            self._logger.info(''.join(traceback.format_exception(*info, limit=5)))
            raise exception

    def branchinitsol(self):
        self.state_buffer = {}

    def branchexeclp(self, allowaddcons):
        self._nb_steps += 1
        previous_reward = self._reward()
        if previous_reward is not None:
            self._return += previous_reward

        state = scip_utilities.extract_state(self.model, self.state_buffer)
        # convert state to tensors
        c, e, v = state
        state = (
            tf.convert_to_tensor(c['values'], dtype=tf.float32),
            tf.convert_to_tensor(e['indices'], dtype=tf.int32),
            tf.convert_to_tensor(e['values'], dtype=tf.float32),
            tf.convert_to_tensor(v['values'], dtype=tf.float32),
            tf.convert_to_tensor([c['values'].shape[0]], dtype=tf.int32),
            tf.convert_to_tensor([v['values'].shape[0]], dtype=tf.int32),
        )
        var_logits = self.actor(state, tf.convert_to_tensor(False)).numpy().squeeze(0)

        candidate_vars, *_ = self.model.getLPBranchCands()
        candidate_mask = [var.getCol().getLPPos() for var in candidate_vars]
        var_logits = var_logits[candidate_mask]

        policy = softmax(var_logits)
        action = np.random.choice(len(policy), 1, p=policy)[0]
        best_var = candidate_vars[action]

        self.model.branchVar(best_var)
        result = pyscipopt.SCIP_RESULT.BRANCHED
        return {"result": result}

    def save_results(self, model, recorder, instance_path, solving_stats_output_dir):
        # Save benchmark
        if instance_path not in self._benchmark:
            self._benchmark[instance_path] = {'return': [], 'nb_nodes': [], 'nb_lp_iterations': [], 'solving_time': []}
        self._benchmark[instance_path]['return'].append(self._return)
        self._benchmark[instance_path]['nb_nodes'].append(model.getNNodes())
        self._benchmark[instance_path]['nb_lp_iterations'].append(model.getNLPIterations())
        self._benchmark[instance_path]['solving_time'].append(model.getSolvingTime())
        with (solving_stats_output_dir/"benchmark.pkl").open("wb") as file:
            pickle.dump(self._benchmark, file)

        # Save solving stats samples
        if solving_stats_output_dir is not None and recorder.stats and self._sample_count < self.nb_solving_stats_samples:
            nb_subsamples = np.ceil(0.05 * len(recorder.stats)).astype(int)
            subsample_ends = np.random.choice(np.arange(1, len(recorder.stats)+1), nb_subsamples, replace=False).tolist()
            for subsample_end in subsample_ends:
                subsample_stats, open_node_stats = scip_utilities.pack_stats(recorder.stats[:subsample_end], recorder.open_node_stats[:subsample_end])
                return_left = self._return - recorder.return_[subsample_end-1]
                nb_nodes_left = model.getNNodes() - recorder.nb_nodes[subsample_end-1]
                nb_lp_iterations_left = model.getNLPIterations() - recorder.nb_lp_iterations[subsample_end-1]
                solving_time_left = model.getSolvingTime() - recorder.solving_time[subsample_end-1]

                if self._sample_count < self.nb_solving_stats_samples:
                    self._sample_count += 1
                    sample_path = solving_stats_output_dir/f"sample_{self._sample_count-1}.pkl"
                    if self._sample_count % 10 == 1:
                        self._logger.info(f"Saving {sample_path}")
                    with gzip.open(str(sample_path), 'wb') as file:
                        pickle.dump({'solving_stats': subsample_stats,
                                     'open_node_stats': open_node_stats['features'],
                                     'mask': open_node_stats['mask'],
                                     'return_left': return_left,
                                     'nb_nodes_left': nb_nodes_left,
                                     'nb_lp_iterations_left': nb_lp_iterations_left,
                                     'solving_time_left': solving_time_left,
                                     'instance_path': instance_path}, file)

    def branchexitsol(self):
        self._reward.snapshot_reward()

    def load_actor(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        tfconfig = tf.ConfigProto()
        tfconfig.intra_op_parallelism_threads = 1
        tfconfig.inter_op_parallelism_threads = 1
        tfconfig.use_per_session_threads = False
        tf.enable_eager_execution(tfconfig)
        tf.set_random_seed(seed=self.seed)

        self._actor_weights = GCNPolicy()
        self._actor_weights.restore_state(self.parameters_path)

    def configure_logger(self):
        self._logger = logging.getLogger("sampler")
        self._logger.setLevel(logging.DEBUG)
        os.makedirs("logs/", exist_ok=True)
        file_handler = logging.FileHandler(f"logs/sampler-{self.id}.log", 'w', 'utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)-8s]  %(message)s',
                                      datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)


class SolvingStatsRecorder(pyscipopt.Eventhdlr):
    """
    A SCIP event handler that records solving stats
    """
    def __init__(self, sampler):
        self.sampler = sampler

        self.stats = []
        self.open_node_stats = []
        self.return_ = []
        self.nb_nodes = []
        self.nb_lp_iterations = []
        self.solving_time = []

    def eventinit(self):
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEFEASIBLE, self)
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEINFEASIBLE, self)
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEBRANCHED, self)

    def eventexit(self):
        self.model.dropEvent(pyscipopt.SCIP_EVENTTYPE.NODEFEASIBLE, self)
        self.model.dropEvent(pyscipopt.SCIP_EVENTTYPE.NODEINFEASIBLE, self)
        self.model.dropEvent(pyscipopt.SCIP_EVENTTYPE.NODEBRANCHED, self)

    def eventexec(self, event):
        if len(self.stats) < self.model.getNNodes():
            self.stats.append(self.model.getSolvingStats())
            self.open_node_stats.append(self.model.getOpenNodeStats())
            self.return_.append(float(self.sampler._return))
            self.nb_nodes.append(self.model.getNNodes())
            self.nb_lp_iterations.append(self.model.getNLPIterations())
            self.solving_time.append(self.model.getSolvingTime())


class Message(enum.Enum):
    NEW_INSTANCE = enum.auto()
    INSTANCE_FINISHED = enum.auto()
    STOP = enum.auto()


class NbNodesRewards:
    def __init__(self, model):
        self.model = model
        self.previous_nb_nodes = None
        self.reward = None

    def __call__(self):
        if self.reward is None:
            self.snapshot_reward()
        reward = self.reward
        self.reward = None
        return reward

    def snapshot_reward(self):
        nb_nodes = self.model.getNNodes()
        if self.previous_nb_nodes is not None:
            self.reward = float(self.previous_nb_nodes - nb_nodes)
            self.previous_nb_nodes = nb_nodes
        else:
            self.reward = None
            self.previous_nb_nodes = self.model.getNNodes()
