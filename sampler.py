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
from config import *


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
            model.freeProb()
            self.results_queue.put({'nb_nodes_total': nb_nodes_total, 
                                    'solving_stats': recorder.stats})
        self.results_queue.put(None)
    
    def branchinit(self):
        self.state_buffer = {}
    
    def branchexeclp(self, allowaddcons):
        state = scip_utilities.extract_state(self.model, self.state_buffer)
        # convert state to tensors
        c, e, v = state
        state = (
            tf.convert_to_tensor(c['values'], dtype=tf.float32),
            tf.convert_to_tensor(e['indices'], dtype=tf.int32),
            tf.convert_to_tensor(e['values'], dtype=tf.float32),
            tf.convert_to_tensor(v['values'], dtype=tf.float32),
        )
        var_logits = self.actor(state).numpy().squeeze(0)

        candidate_vars, *_ = self.model.getLPBranchCands()
        branching_mask = [var.getCol().getLPPos() for var in candidate_vars]
        var_logits = var_logits[branching_mask]
        best_var = candidate_vars[var_logits.argmax()]

        self.model.branchVar(best_var)
        result = pyscipopt.SCIP_RESULT.BRANCHED
        return {"result": result}

    def load_actor(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        tfconfig.log_device_placement = True
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


class Message(enum.Enum):
    NEW_INSTANCE = enum.auto()
    INSTANCE_FINISHED = enum.auto()
    STOP = enum.auto()
