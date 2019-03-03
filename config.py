"""
Hyperparameters for the critic model.
"""

SEQUENCE_LENGTH = 50
SOLVING_STATS_FEATURES = [
'opennodes_90quant_norm',
'opennodes_75quant_normfirst',
'opennodes_90quant_normfirst',
'cutoffbound',
'avgpseudocostscorecurrentrun',
'primalbound',
'nnodelpiterations',
'dualboundroot',
'ndeactivatednodes',
'ncreatednodesrun',
'ntotalnodes',
'nleaves',
'nduallps',
'nstrongbranchs',
'nlps',
'nnodelps',
'nnodeinitlpiterations',
'gap',
'avgpseudocostscore_normfirst',
'nnodes_done',
'nnodesleft',
'transgap',
'nbacktracks',
'avgdualbound_normfirst',
'avgpseudocostscore_norm',
]