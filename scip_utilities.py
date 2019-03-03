"""
Miscellaneous SCIP-related utilities.
"""
import numpy as np


def init_scip_params(model, seed, heuristics=True):
    """
    Initialize a SCIP model.
    
    Parameters
    ----------
    seed : int
        The SCIP seed.
    heuristics : bool
        Should heuristics be activated?
    """
    seed = seed % 2147483648  # SCIP seed range

    # set up randomization
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', seed)
    model.setIntParam('randomization/randomseedshift', seed)

    # disable presolving and separating (node cuts)
    model.setIntParam('presolving/maxrounds', 0)
    model.setIntParam('presolving/maxrestarts', 0)
    model.setIntParam('separating/maxrounds', 0)
    model.setIntParam('separating/maxroundsroot', 0)

    # disable conflict analysis (more cuts)
    model.setBoolParam('conflict/enable', False)

    # if asked, disable all heuristics
    if not heuristics:
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)


def extract_state(model, buffer=None):
    """
    Compute a bipartite graph representation of the solver. In this
    representation, the variables and constraints of the MILP are the
    left- and right-hand side nodes, and an edge links two nodes iff the
    variable is involved in the constraint. Both the nodes and edges carry
    features.

    Parameters
    ----------
    model : pyscipopt.scip.Model
        The current model.
    buffer : dict
        A buffer to avoid re-extracting redundant information from the solver
        each time.
    
    Returns
    -------
    variable_features : dictionary of type {'names': list, 'values': np.ndarray}
        The features associated with the variable nodes in the bipartite graph.
    edge_features : dictionary of type ('names': list, 'indices': np.ndarray, 'values': np.ndarray}
        The features associated with the edges in the bipartite graph.
        This is given as a sparse matrix in COO format.
    constraint_features : dictionary of type {'names': list, 'values': np.ndarray}
        The features associated with the constraint nodes in the bipartite graph.
    """
    if buffer is None:
        buffer = {}

    # update state from buffer if any
    s = model.getState(buffer['scip_state'] if 'scip_state' in buffer else None)

    if 'np_state' not in buffer:
        obj_norm = np.linalg.norm(s['col']['coefs'])
        obj_norm = 1 if obj_norm <= 0 else obj_norm
        row_norms = s['row']['norms']
        row_norms[row_norms == 0] = 1
        # assume only LHS or RHS type rows
        assert all(np.isnan(s['row']['lhss'])) or all(np.isnan(s['row']['rhss']))
        row_dir_lhs = all(np.isnan(s['row']['rhss']))
    else:
        constraint_features, edge_features, variable_features, tmp = buffer['np_state']
        obj_norm = tmp['obj_norm']
        row_norms = tmp['row_norms']
        row_dir_lhs = tmp['row_dir_lhs']

    # Column features
    n_cols = len(s['col']['types'])
    col_feats = {}

    if 'np_state' not in buffer:
        col_feats['type'] = np.zeros((n_cols, 4))  # BINARY INTEGER IMPLINT CONTINUOUS
        col_feats['type'][np.arange(n_cols), s['col']['types']] = 1
        col_feats['coef_normalized'] = s['col']['coefs'].reshape(-1, 1) / obj_norm

    col_feats['has_lb'] = ~np.isnan(s['col']['lbs']).reshape(-1, 1)
    col_feats['has_ub'] = ~np.isnan(s['col']['ubs']).reshape(-1, 1)
    col_feats['sol_is_at_lb'] = s['col']['sol_is_at_lb'].reshape(-1, 1)
    col_feats['sol_is_at_ub'] = s['col']['sol_is_at_ub'].reshape(-1, 1)
    col_feats['sol_frac'] = s['col']['solfracs'].reshape(-1, 1)
    col_feats['sol_frac'][s['col']['types'] == 3] = 0  # continuous have no fractionality
    col_feats['basis_status'] = np.zeros((n_cols, 4))  # LOWER BASIC UPPER ZERO
    col_feats['basis_status'][np.arange(n_cols), s['col']['basestats']] = 1
    col_feats['reduced_cost'] = s['col']['redcosts'].reshape(-1, 1) / obj_norm
    col_feats['age'] = s['col']['ages'].reshape(-1, 1) / (s['stats']['nlps'] + 5)
    col_feats['sol_val'] = s['col']['solvals'].reshape(-1, 1)
    col_feats['inc_val'] = s['col']['incvals'].reshape(-1, 1)
    col_feats['avg_inc_val'] = s['col']['avgincvals'].reshape(-1, 1)

    col_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in col_feats.items()]
    col_feat_names = [n for names in col_feat_names for n in names]
    col_feat_vals = np.concatenate(list(col_feats.values()), axis=-1)

    # if no buffer, create features
    if 'np_state' not in buffer:
        variable_features = {'names': col_feat_names, 'values': col_feat_vals,}
    # else update feature values
    else:
        col_feat_inds = [variable_features['names'].index(feat) for feat in col_feat_names]
        variable_features['values'][:, col_feat_inds] = col_feat_vals

    # Row features
    n_rows = len(s['row']['nnzrs'])
    row_feats = {}

    if 'np_state' not in buffer:
        row_feats['obj_cosine_similarity'] = s['row']['objcossims'].reshape(-1, 1)
        if row_dir_lhs:
            row_feats['obj_cosine_similarity'] *= -1
            row_feats['bias'] = (-s['row']['lhss'] / row_norms).reshape(-1, 1)
        else:
            row_feats['bias'] = (s['row']['rhss'] / row_norms).reshape(-1, 1)

    row_feats['is_tight'] = s['row']['is_at_lhs'].reshape(-1, 1)
    row_feats['dualsol_val_normalized'] = (s['row']['dualsols'] / (row_norms * obj_norm)).reshape(-1, 1)
    row_feats['dualsol_val_normalized'] /= np.linalg.norm(row_feats['dualsol_val_normalized'])
    row_feats['basis_status'] = np.zeros((n_rows, 4))  # LOWER BASIC UPPER ZERO
    row_feats['basis_status'][np.arange(n_rows), s['row']['basestats']] = 1
    row_feats['age'] = s['row']['ages'].reshape(-1, 1) / (s['stats']['nlps'] + 5)

    row_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in row_feats.items()]
    row_feat_names = [n for names in row_feat_names for n in names]
    row_feat_vals = np.concatenate(list(row_feats.values()), axis=-1)

    # if no buffer, create features
    if 'np_state' not in buffer:
        constraint_features = {
            'names': row_feat_names,
            'values': row_feat_vals,}
    # else update feature values
    else:
        row_feat_inds = [constraint_features['names'].index(feat) for feat in row_feat_names]
        constraint_features['values'][:, row_feat_inds] = row_feat_vals

    # Edge features
    # if no buffer, create features
    if 'np_state' not in buffer:
        edge_row_idxs, edge_col_idxs = s['nzrcoef']['rowidxs'], s['nzrcoef']['colidxs']
        edge_feats = {}

        edge_feats['coef_normalized'] = (s['nzrcoef']['vals'] / row_norms[edge_row_idxs]).reshape(-1, 1)
        if row_dir_lhs:
            edge_feats['coef_normalized'] *= -1

        edge_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in edge_feats.items()]
        edge_feat_names = [n for names in edge_feat_names for n in names]
        edge_feat_indices = np.vstack([edge_row_idxs, edge_col_idxs])
        edge_feat_vals = np.concatenate(list(edge_feats.values()), axis=-1)

        edge_features = {
            'names': edge_feat_names,
            'indices': edge_feat_indices,
            'values': edge_feat_vals,}
    # else update feature values
    else:
        pass

    buffer['scip_state'] = s
    buffer['np_state'] = (
        constraint_features,
        edge_features,
        variable_features,
        {
            'obj_norm': obj_norm,
            'row_norms': row_norms,
            'row_dir_lhs': row_dir_lhs,
        },
    )

    return constraint_features, edge_features, variable_features


SOLVING_STATS_SEQUENCE_LENGTH = 50
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


def pack_solving_stats(solving_stats):
    solving_stats = {name: np.asarray([s[name]
                           for s in solving_stats[-SOLVING_STATS_SEQUENCE_LENGTH:]]) 
                           for name in solving_stats[0].keys()}
    solving_stats = normalize_solving_stats(solving_stats, 
                              length=SOLVING_STATS_SEQUENCE_LENGTH)
    solving_stats = np.stack([solving_stats[feature_name] 
                              for feature_name in SOLVING_STATS_FEATURES], axis=-1)
    return solving_stats


def normalize_solving_stats(solving_stats, length=SOLVING_STATS_SEQUENCE_LENGTH):
    solving_stats = {name: np.pad(vals[-length:], (max(length-len(vals), 0), 0), mode='edge') for name, vals in solving_stats.items()}
    
    nnodes_done = solving_stats['ninternalnodes'] + solving_stats['nfeasibleleaves'] + solving_stats['ninfeasibleleaves'] + solving_stats['nobjlimleaves']
    solving_stats['nnodes_done'] = nnodes_done
    lp_obj_norm = [(v - lb) / ((ub - lb) if ub > lb else 1) for v, lb, ub in zip(solving_stats['lp_obj'], solving_stats['dualbound'], solving_stats['primalbound'])]
    solving_stats['lp_obj_norm'] = lp_obj_norm
    lp_obj_normfirst = [(v - solving_stats['dualbound'][0]) / ((solving_stats['primalbound'][0] - solving_stats['dualbound'][0]) if solving_stats['primalbound'][0] > solving_stats['dualbound'][0] else 1) for v in solving_stats['lp_obj']]
    solving_stats['lp_obj_normfirst'] = lp_obj_normfirst
    avgdualbound_normfirst = [(v - solving_stats['dualbound'][0]) / ((solving_stats['primalbound'][0] - solving_stats['dualbound'][0]) if solving_stats['primalbound'][0] > solving_stats['dualbound'][0] else 1) for v in solving_stats['avgdualbound']]
    solving_stats['avgdualbound_normfirst'] = avgdualbound_normfirst
    avgpseudocostscore_norm = [(v - lb) / ((ub - lb) if ub > lb else 1) for v, lb, ub in zip(solving_stats['avgpseudocostscore'], solving_stats['dualbound'], solving_stats['primalbound'])]
    solving_stats['avgpseudocostscore_norm'] = avgpseudocostscore_norm
    avgpseudocostscore_normfirst = [(v - solving_stats['dualbound'][0]) / ((solving_stats['primalbound'][0] - solving_stats['dualbound'][0]) if solving_stats['primalbound'][0] > solving_stats['dualbound'][0] else 1) for v in solving_stats['avgpseudocostscore']]
    solving_stats['avgpseudocostscore_normfirst'] = avgpseudocostscore_normfirst              
    for k in (10, 25, 50, 75, 90):
        quint = f'opennodes_{k}quant'
        quint_norm = f'opennodes_{k}quant_norm'
        quint_normfirst = f'opennodes_{k}quant_normfirst'
        opennodes_quint_norm = [(v - lb) / ((ub - lb) if ub > lb else 1) for v, lb, ub in zip(solving_stats[quint], solving_stats['dualbound'], solving_stats['primalbound'])]
        solving_stats[quint_norm] = opennodes_quint_norm
        opennodes_quint_normfirst = [(v - solving_stats['dualbound'][0]) / ((solving_stats['primalbound'][0] - solving_stats['dualbound'][0]) if solving_stats['primalbound'][0] > solving_stats['dualbound'][0] else 1) for v in solving_stats[quint]]
        solving_stats[quint_normfirst] = opennodes_quint_normfirst
    return solving_stats


