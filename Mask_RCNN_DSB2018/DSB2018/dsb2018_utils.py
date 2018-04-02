import numpy as np


def ismember(a, b, index_requested = True):
    """
    Function that takes two input arrays (ndim = 1)
    and returns a boolean array of a.shape of True if a is in b and False if not.
    If index_requested = True then the row of b = a is returned
    """  

    a, reverse_idx = np.unique(a, return_inverse=True) # reverse_idx = the indices of the unique array a that result in the original array a
    b, b_idx = np.unique(b, return_index = True) # b_idx = the indices of b that result in the unique array b
    
    # Method:
    # 1. Concatenate unique values
    unique_a_b = np.concatenate((a, b))     

    # 2. Sort them (stable sort) so that a's value is followed by b's value if they are the same
    order = unique_a_b.argsort(kind='mergesort')
    sorted_unique_a_b = unique_a_b[order]      
    bool_a_in_b = (sorted_unique_a_b[1:] == sorted_unique_a_b[:-1]) 
    ismember_flag = np.concatenate((bool_a_in_b, [False]))  # Flag for a same as b

    # 3. Create a_in_b boolean array (based on the index of a)
    a_in_b = np.empty(unique_a_b.shape, dtype=bool)
    a_in_b[order] = ismember_flag

    # Walk through the same steps if the index of b for each a in b is requested 
    if index_requested:
        b_idx = np.concatenate((np.ones(a.shape, dtype = np.int64) * -1, b_idx))
        sorted_b_idx = b_idx[order]
        ismember_flag_idx = np.concatenate(([False], bool_a_in_b))
        a_in_b_idx = np.ones(unique_a_b.shape, dtype = np.int64) * -1
        a_in_b_idx[a_in_b] = sorted_b_idx[ismember_flag_idx]

        return a_in_b[reverse_idx], a_in_b_idx[reverse_idx]

    return a_in_b[reverse_idx]


def stripback(x):
    """
    if x is surrounded by a superfulous wrapper, strip it back
    """
    if x is None or unknown_dtype(x):
        return x

    stripped = False
    while not stripped:
        if not isinstance(x, list):
            stripped = True
        elif len(x) > 1:
            stripped = True
        else:
            x = x[0]

    if isinstance(x, list):
        x = [stripback(_x) for _x in x]

    return x


def reduce_dict(this_dict, apply_index):

    def can_apply(v, apply_index):

        apply = False

        if is_numpy(apply_index) and hasattr(apply_index, '__len__'):
            if apply_index.dtype == np.bool:
                apply = (is_numpy(v) and v.shape[0] == len(apply_index))
            else:
                apply = (is_numpy(v) and v.shape[0] > max(apply_index))
        else:
            apply = (is_numpy(v) and v.shape[0] > apply_index)

        return apply

    if isinstance(this_dict, dict):
        return {k: reduce_dict(v, apply_index) if isinstance(v, (dict, list)) else v[apply_index] if can_apply(v, apply_index) else v for k, v in this_dict.items()}
    elif isinstance(this_dict, list):
        return [reduce_dict(d, apply_index) for d in this_dict]
    elif is_numpy(this_dict):
        return this_dict[apply_index]
    else:
        return this_dict


def concatenate_list_of_dicts(list_of_dicts):

    # Make sure everything is in the correct format
    list_of_dicts = [stripback(d) for d in list_of_dicts]

    if any([not isinstance(d, dict) for d in list_of_dicts]): 
        return list_of_dicts

    concat_dict = {}
    for key in list_of_dicts[0].keys():
        if isinstance(list_of_dicts[0][key], dict):
            concat_dict[key] = concatenate_list_of_dicts([d[key] for d in list_of_dicts])
        elif isinstance(list_of_dicts[0][key], list):
            try:
                # Non jagged entries
                concat_dict[key] = [np.concatenate([np.asarray(d[key][i]) if np.asarray(d[key][i]).ndim > 1 else np.asarray(d[key][i]).reshape(-1, 1) for d in list_of_dicts]) for i in range(len(list_of_dicts[0][key]))]
            except:
                # jagged entries
                concat_dict[key] = jagged_list_to_np([d[key] for d in list_of_dicts])
        else:
            try:
                concat_dict[key] = np.concatenate([np.asarray(d[key]) if np.asarray(d[key]).ndim > 1 else np.asarray(d[key]).reshape(-1, 1) for d in list_of_dicts])
            except:
                concat_dict[key] = jagged_list_to_np([d[key] for d in list_of_dicts])

    return concat_dict
    

def unknown_dtype(obj):
    return not isinstance(obj, (dict, list, int, float, str)) and not is_numpy(obj)

def is_numpy(obj):
    return type(obj).__module__ == np.__name__


def jagged_list_to_np(x): 

    output = np.empty((len(x),), dtype = object)

    for i, _x in enumerate(x):
        output[i] = _x

    return output
