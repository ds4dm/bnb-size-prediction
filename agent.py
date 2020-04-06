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
import threading
import pyscipopt
import scip_utilities
# from wurlitzer import sys_pipes
from scip import Scip, ScipMessage
from actor.model import GCNPolicy
from scipy.special import softmax
from pathlib import Path

logger = logging.getLogger("rl2branch." + __name__)


class AsyncAgent(threading.Thread):
    def __init__(self,inQueue,outQueue,reward_type,greedy=False,record_tseries=False):
        super().__init__()
        self.scip = Scip(name=f"{self.name}'s SCIP",reward_type=reward_type)
        self.actor = None
        self.inQueue = inQueue
        self.outQueue = outQueue
        self.greedy_mode = greedy
        self.record_tseries = record_tseries

        self.finished = False
        self.instance = None
        self.instance_finished = threading.Event()
        self.stats = None
        self.must_stop = False
        self.random = np.random.RandomState(0)
        self.scip.start()

    def pass_actor_ref(self,actor):
        self.actor = actor

    def run(self):
        try:
            logger.info("Started agent")
            while not self.inQueue.empty():
                instance, seed = self.inQueue.get()
                name = str(instance).split('/')[-1]
                logger.info(f"Solving {name} with seed {seed}")
                self._load_instance(instance, seed)
                while not self.instance_finished.is_set():
                    self.step()
                self.outQueue.put(self.stats)
        finally:
            self.finished = True
            logger.info("Finished")
            self.scip.tell({'type': ScipMessage.KILL})
            self.scip.schedule_instance((None,None))
            self.scip.flush_message_pipe()
            self.scip.join()

    def step(self):
        """
        Run a single step through the instance.

        Parameters
        ----------
        memory : EpisodeMemory
            The memory to save transitions to.
        """
        message = self.scip.listen()
        if message is not None:
            if message['type'] == ScipMessage.ACTION_NEEDED:
                constraint_features, edge_features, variable_features = message['state']
                candidate_mask = message['candidate_mask']

                constraint_features = tf.convert_to_tensor(constraint_features['values'], tf.float32)
                edge_indices = tf.convert_to_tensor(edge_features['indices'], tf.int32)
                edge_features = tf.convert_to_tensor(edge_features['values'], tf.float32)
                variable_features = tf.convert_to_tensor(variable_features['values'], tf.float32)
                n_cons = tf.convert_to_tensor([constraint_features.shape[0]],dtype=tf.int32)
                n_vars = tf.convert_to_tensor([variable_features.shape[0]],dtype=tf.int32)
                state = constraint_features, edge_indices, edge_features, variable_features, n_cons, n_vars
                candidate_mask = tf.convert_to_tensor(candidate_mask, tf.float32)

                log_policy = self.actor(state,tf.convert_to_tensor(False))
                log_policy = log_policy - tf.stop_gradient(tf.reduce_max(log_policy, axis=-1, keepdims=True))
                policy = tf.exp(log_policy) * candidate_mask
                policy = policy / tf.expand_dims(tf.reduce_sum(policy, axis=-1), axis=-1)

                if self.greedy_mode:
                    action = tf.argmax(policy, axis=-1)
                    logprob_action = tf.zeros((1,))
                else:
                    action = tf.convert_to_tensor(self.random.choice(policy.shape[-1], 1, p=policy[0, :].numpy()))
                    logprob_action = tf.log(policy[:, action[0]] + 1e-5)

                self.scip.tell({'type': ScipMessage.ACTION,
                                'action': action[0].numpy()})

            elif message['type'] == ScipMessage.INSTANCE_FINISHED:
                self.stats['nb_nodes'] = message['nb_nodes']
                self.stats['nb_lp_iterations'] = message['nb_lp_iterations']
                self.stats['solving_time'] = message['solving_time']
                if self.record_tseries:
                    self.stats['time_series'] = message['tseries']
                self.instance_finished.set()
                logger.info("Instance finished")

            elif message['type'] == ScipMessage.EXCEPTION_THROWN:
                raise message['exception']

            else:
                raise ValueError(f"Unrecognized message type {message['type']}")


        if self.must_stop:
            logger.info("Received termination signal")
            self.scip.tell({'type': ScipMessage.STOP})
            self.instance_finished.set()
            self.must_stop = False

    def _load_instance(self, instance, seed):
        """
        Load an instance.

        Parameters
        ----------
        instance : str
            Path to the instance.
        """
        self.scip.schedule_instance((instance,seed))
        self.instance_finished.clear()
        self.instance = instance
        self.stats = {'instance': instance, 'seed': seed,
                      'nb_nodes' : None, 'nb_lp_iterations' : None,
                      'solving_time' : None, 'time_series': None}

    def stop(self):
        """
        Stop the solving process of the current instance.
        """
        self.must_stop = True
