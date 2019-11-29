"""
Code that deals with SCIP, the underlying solver that serves as environment.
"""

import sys
import queue
import enum
import multiprocessing as mp
import traceback
import numpy as np
import pyscipopt
import scip_utilities
import rewards
from wurlitzer import sys_pipes
from utilities import internal, external


class Scip(mp.Process, pyscipopt.Branchrule):
    """
    A SCIP environment for reinforcement learning.

    Parameters
    ----------
    name : str, optional
        A name for the process. If None, will use the default from
        the multiprocessing library.
    """
    def __init__(self, *_, name=None,reward_type):
        super().__init__(name=name)
        self._instance_to_scip, self._instance_from_parent = mp.Pipe(duplex=False)
        self._message_to_scip, self._message_from_parent = mp.Pipe(duplex=False)
        self._message_to_parent, self._message_from_scip = mp.Pipe(duplex=False)
        self._reward_type = reward_type
        self.reward = None
        self.data_recorder = None
        self.process_must_end = False

    def start(self):
        super().start()
        self._message_from_scip.close()

    def run(self):
        """
        Main SCIP function.
        """
        try:
            while not self.process_must_end:
                instance_file, seed = self._instance_to_scip.recv()
                if instance_file is None:
                    break
                model = pyscipopt.Model()
                model.setIntParam('display/verblevel', 0)
                model.readProblem(instance_file)
                scip_utilities.init_scip_params(model, seed=seed)
                model.includeBranchrule(self, name='ReinforcementBrancher', desc='Reinforcement brancher',
                                        priority=1000000, maxdepth=-1, maxbounddist=1.0)
                self.data_recorder = {'nb_nodes': [], 'nb_lp_iterations': [], 'c_states': [], 'rewards': []}
                self.reward = rewards.getRewardFun(self._reward_type,model)
                with sys_pipes():
                    model.optimize()

                if model.getStatus() == "userinterrupt":
                    break
                else:
                    reward = self.reward()
                    self.data_recorder['rewards'].append(reward)
                    self.tell({'type': ScipMessage.INSTANCE_FINISHED,
                               'reward': reward,
                               'nb_nodes_final': model.getNNodes(),
                               'solving_time_final': model.getSolvingTime(),
                               'nb_lp_iterations_final': model.getNLPIterations(),
                               'gap': model.getGap(),
                               'status': model.getStatus()})
                model.freeProb()

        except Exception as exception:
            self.tell({'type': ScipMessage.EXCEPTION_THROWN,
                       'exception': ScipException(exception=exception, process_name=self.name)})

    def branchinitsol(self):
        """
        Branching initialization procedure, called after model creation, from inside the process
        by pyscipopt.

        """
        self.state_buffer = {}
        self.variables = self.model.getVars(transformed=True)

    def branchexeclp(self, allowaddcons):
        """
        Single step of the branching procedure, called from inside the process by pyscipopt.

        Returns
        -------
        dict
            Dictionary of results expected by pyscipopt.
        """
        try:
            state = scip_utilities.extract_state(self.model, self.state_buffer)
            candidate_mask = self.compute_candidate_mask()
            previous_reward = self.reward()
            self.fetch_stats(previous_reward)
            solving_stats = scip_utilities.pack_solving_stats(self.data_recorder['c_states'])

            self.tell({'type': ScipMessage.ACTION_NEEDED,
                        'previous_reward' : previous_reward,
                        'state' : state,
                        'solving_stats' : solving_stats,
                        'candidate_mask': candidate_mask})

            message = self.listen()
            if message['type'] == ScipMessage.ACTION:
                chosen_variable = self.variables[message['action']]
                self.model.branchVar(chosen_variable)
                sys.stdout.flush()
                return {"result": pyscipopt.SCIP_RESULT.BRANCHED}

            elif message['type'] == ScipMessage.STOP:
                self.model.interruptSolve()
                return {"result": pyscipopt.SCIP_RESULT.DIDNOTRUN}

            elif message['type'] == ScipMessage.KILL:
                self.model.interruptSolve()
                self.process_must_end = True
                return {"result": pyscipopt.SCIP_RESULT.DIDNOTRUN}

            else:
                raise ValueError(f"Unrecognized message type {message['type']}")

        except Exception as exception:
            self.tell({'type': ScipMessage.EXCEPTION_THROWN,
                       'exception': ScipException(exception=exception, process_name=self.name)})
            self.model.interruptSolve()
            return {"result": pyscipopt.SCIP_RESULT.DIDNOTRUN}

    def tell(self, message):
        """
        Send a message.

        Parameters
        ----------
        message : dict
            The message to send.
        """
        if mp.current_process() == self:
            self._message_from_scip.send(message)
        else:
            self._message_from_parent.send(message)

    def listen(self):
        """
        Listen for the most recent message.

        Parameters
        ----------
        timeout : int, optional
            Maximum timeout before giving up listening for a message.

        Returns
        -------
        dict or None
            The most recent message, None if none.
        """
        if mp.current_process() == self:
            endpoint = self._message_to_scip
        else:
            endpoint = self._message_to_parent

        # Process can die while listening: this loop avoids gridlock
        message = None
        while self.is_alive() and message is None:
            try:
                if endpoint.poll(1):
                    message = endpoint.recv()
            except EOFError:
                message = None
        return message

    def flush_message_pipe(self):
        if mp.current_process() == self:
            endpoint = self._message_to_scip
        else:
            endpoint = self._message_to_parent

        exception_ = None
        try:
            while endpoint.poll(1):
                message = endpoint.recv()
                if message['type'] == ScipMessage.EXCEPTION_THROWN:
                    exception_ = message['exception']
        except EOFError:
            pass
        finally:
            if exception_ is not None:
                raise exception_

    @external
    def schedule_instance(self, instance_file):
        self._instance_from_parent.send(instance_file)

    @internal
    def compute_candidate_mask(self):
        """
        Computes a mask over variables with ones on variables we can branch on at the most recent state,
        and zeros where we can't.

        Returns
        -------
        candidates_mask : np.ndarray
            A binary mask that indicates which variables can be branched on.
        """
        candidate_variables, *_ = self.model.getLPBranchCands()
        candidate_indices = np.array([variable.getCol().getLPPos() for variable in candidate_variables])
        candidate_mask = np.zeros((len(self.variables),))
        candidate_mask[candidate_indices] = 1.
        # if self.name == "Agent-1's SCIP":
        #     raise ValueError()

        return candidate_mask

    def fetch_stats(self,previous_reward):
        self.data_recorder['nb_nodes'].append(self.model.getNNodes())
        self.data_recorder['nb_lp_iterations'].append(self.model.getNLPIterations())
        self.data_recorder['c_states'].append(self.model.getSolvingStats())
        if previous_reward is not None:
            self.data_recorder['rewards'].append(previous_reward)

class ScipMessage(enum.Enum):
    ACTION = enum.auto()
    ACTION_NEEDED = enum.auto()
    INSTANCE_FINISHED = enum.auto()
    STOP = enum.auto()
    KILL = enum.auto()
    EXCEPTION_THROWN = enum.auto()


class ScipException(Exception):
    """
    A wrapper around an exception that:
    - contains information about the original exception; and
    - can be sent through a pipe to a different process for processing.

    Parameters
    ----------
    exception : Exception
        The original exception.
    process_name : str
        The name of the process that raised the exception.
    """
    def __init__(self, exception=None, process_name=None):
        super().__init__()
        self.exception = exception
        self.process_name = process_name

        if exception is not None:
            info = type(exception), exception, exception.__traceback__
            self.formatted = ''.join(traceback.format_exception(*info, limit=5))

    def __str__(self):
        return f"(in {self.process_name} process)\n\n{self.formatted}"
