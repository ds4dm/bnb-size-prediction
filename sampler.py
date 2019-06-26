
"""
Code for the actor sampler, for generating datasets for the critic.
"""
import os
import time
import enum
import numpy as np
import multiprocessing as mp
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import pyscipopt
import scip_utilities
# from wurlitzer import sys_pipes
from actor.model import GCNPolicy
from scipy.special import softmax


class ActorSampler(mp.Process, pyscipopt.Branchrule):
    def __init__(self, parameters_path, instance_queue, results_queue, seed):
        super().__init__()
        self.parameters_path = parameters_path
        self.instance_queue = instance_queue
        self.results_queue = results_queue
        self.seed = seed
        self.actor = None
    
    def run(self):
        self.load_actor()
        while True:
            message = self.instance_queue.get()
            if message['type'] == Message.NEW_INSTANCE:
                instance_path = message['instance_path']
            elif message['type'] == Message.STOP:
                break
            else:
                raise ValueError(f"Unrecognized message {message}")

            model = pyscipopt.Model()
            model.setIntParam('display/verblevel', 0)
            model.readProblem(instance_path)
            scip_utilities.init_scip_params(model, seed=self.seed)

            recorder = SolvingStatsRecorder()
            model.includeEventhdlr(recorder, "SolvingStatsRecorder", "")
            model.includeBranchrule(branchrule=self,
                name="My branching rule", desc="",
                priority=666666, maxdepth=-1, maxbounddist=1)
            
            model.optimize()
            nb_nodes_total = model.getNNodes()
            nb_lp_iterations_total = model.getNLPIterations()
            solving_time_total = model.getSolvingTime()
            model.freeProb()
            self.results_queue.put({'nb_nodes_total': nb_nodes_total, 
                                    'nb_lp_iterations_total': nb_lp_iterations_total, 
                                    'solving_time_total': solving_time_total, 
                                    'solving_stats': recorder.stats,
                                    'nb_nodes': recorder.nb_nodes,
                                    'nb_lp_iterations': recorder.nb_lp_iterations,
                                    'solving_time': recorder.solving_time,
                                    'instance_path': instance_path})
        self.results_queue.put(None)
    
    def branchinit(self):
        self.state_buffer = {}
    
    def branchexeclp(self, allowaddcons):
        print(f"  {self.name}: Extracting state...", end="")
        state = scip_utilities.extract_state(self.model, self.state_buffer)
        print(f" done!")
        # convert state to tensors
        c, e, v = state
        state = (
            tf.convert_to_tensor(c['values'], dtype=tf.float32),
            tf.convert_to_tensor(e['indices'], dtype=tf.int32),
            tf.convert_to_tensor(e['values'], dtype=tf.float32),
            tf.convert_to_tensor(v['values'], dtype=tf.float32),
        )
        nb_constraints = tf.convert_to_tensor([c['values'].shape[0]], dtype=tf.int32)
        nb_variables = tf.convert_to_tensor([v['values'].shape[0]], dtype=tf.int32)
        var_logits = self.actor((*state, nb_constraints, nb_variables)).numpy().squeeze(0)

        candidate_vars, *_ = self.model.getLPBranchCands()
        candidate_mask = [var.getCol().getLPPos() for var in candidate_vars]
        var_logits = var_logits[candidate_mask]

        policy = softmax(var_logits)
        action = np.random.choice(len(policy), 1, p=policy)[0]
        best_var = candidate_vars[action]

        self.model.branchVar(best_var)
        result = pyscipopt.SCIP_RESULT.BRANCHED
        return {"result": result}

    def load_actor(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        tfconfig = tf.ConfigProto()
        tf.enable_eager_execution(tfconfig)
        tf.set_random_seed(seed=self.seed)

        actor = GCNPolicy()
        actor.restore_state(self.parameters_path)
        self.actor = tfe.defun(actor.call)

    
class SolvingStatsRecorder(pyscipopt.Eventhdlr):
    """
    A SCIP event handler that records solving stats
    """
    def __init__(self):
        self.stats = []
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
            self.nb_nodes.append(self.model.getNNodes())
            self.nb_lp_iterations.append(self.model.getNLPIterations())
            self.solving_time.append(self.model.getSolvingTime())


class Message(enum.Enum):
    NEW_INSTANCE = enum.auto()
    INSTANCE_FINISHED = enum.auto()
    STOP = enum.auto()
