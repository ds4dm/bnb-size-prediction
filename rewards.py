class NbNodesReward:
    def __init__(self, model):
        self.model = model
        self.previous_nb_nodes = None

    def __call__(self):
        nb_nodes = self.model.getNNodes()
        if self.previous_nb_nodes is not None:
            reward = float(self.previous_nb_nodes - nb_nodes)
            self.previous_nb_nodes = nb_nodes
        else:
            reward = None
            self.previous_nb_nodes = self.model.getNNodes()
        return reward

    def cleanup(self):
        self.previous_nb_nodes = None


# No other reward type can be used until the problem with
# the critic timing is solved!!

# class GapReward:
#     def __init__(self,model):
#         self.model = model
#         self.gap0 = None
#         self.prev_gap = None
#
#     def __call__(self):
#         incumbent = self.model.getUpperbound()
#         best = self.model.getLowerbound()
#         gap = incumbent - best
#
#         assert incumbent <1e20
#
#         if self.gap0 == None:
#             self.gap0 = gap
#             self.prev_gap = self.gap0
#             reward = None
#         elif gap == self.prev_gap:
#             reward = -1.0
#         else:
#             self.prev_gap = gap
#             reward = -1.0 + (self.gap0 - gap) / self.gap0
#
#         return reward
#
#     def cleanup(self):
#         self.gap0 = None
#         self.prev_gap = None

def getRewardFun(type,model):
    if type == 'nbnodes':
        return NbNodesReward(model)
    #elif type == 'gap':
    #    return GapReward(model)
    else:
        raise ValueError('Invalid reward type')
