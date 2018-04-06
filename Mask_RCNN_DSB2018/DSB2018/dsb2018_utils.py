import numpy as np
from utils import *


def create_id(x_in):
    """
    Function that takes in a matrix of values (x_in)
    and converts these to ids, starting from 0
    """
    (u_x_in, idx) = unique_rows(x_in)
    return idx.astype(np.int64)


def unique_rows(x_in):
    """
    Function that takes in an ndarray of values (x_in)
    and returns unique rows aswell as the indices 
    which those rows occur in
    """
    if len(x_in.shape) > 1:
        x_in_transform = combine_rows(x_in)
    else:
        x_in_transform = x_in

    u_x_in_transform, idx, u_x_in_idx = np.unique(x_in_transform, return_index = True, return_inverse = True)
    u_x_in = x_in[idx]
    return (u_x_in, u_x_in_idx)


def combine_rows(x_in):
    """ 
    Function that transforms a 2-dimensional array, x_in,
    into a single dimension.
    """
    #return np.asanyarray(x_in.view([('', x_in.dtype)]*x_in.shape[1])).flatten()
    try:
        return np.asanyarray(x_in.view(np.dtype((np.void, x_in.dtype.itemsize * x_in.shape[1])))).flatten() # This one seems to work much quicker in subsequent functions  
    except:
        # see https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.view.html
        z = x_in.copy()     
        return np.asanyarray(z.view(np.dtype((np.void, z.dtype.itemsize * z.shape[1])))).flatten() # This one seems to work


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


def maskrcnn_mask_to_labels(masks):
    """
    Converts masks to labels.
    """
    mask_shape = masks.shape[:2]

    labels = np.zeros(mask_shape, dtype = np.int)
    for i, mask in enumerate(list(np.moveaxis(masks, -1, 0))):
        labels[mask > 0] = (i + 1)

    # Make sure they are relabeled to go from 1 -> N
    labels = create_id(labels.flatten()).reshape(mask_shape)

    return labels


def maskrcnn_labels_to_mask(labels):
    """
    Converts labels to masks
    """
    mask = np.stack([(labels == i).astype(np.int) for i in range(1, int(labels.max()) + 1)], axis = -1)

    return mask


def run_length_decode(rel, H, W, fill_value = 255, index_offset = 0):
    mask = np.zeros((H * W), np.uint8)
    if rel != '':
        rel  = np.array([int(s) for s in rel.split(' ')]).reshape(-1, 2)
        for r in rel:
            start = r[0] - index_offset
            end   = start + r[1]
            mask[start : end] = fill_value
    mask = mask.reshape(H, W)
    return mask


def labels_from_rles(mask_rles, mask_shape):

    masks = [run_length_decode(rle, mask_shape[1], mask_shape[0], 1, index_offset = 1).T for rle in mask_rles]
    labels = np.zeros(mask_shape, dtype = np.int)

    for i, mask in enumerate(masks):
        labels += (mask * (i + 1))

    return labels, masks


def combine_boxes(boxes, scores, masks, threshold):
    """
    Combines boxes if their IOU is above threshold.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes 
    ixs = np.arange(boxes.shape[0])
    n_joins = np.ones(ixs.shape)
    ixs_pick = []

    while len(ixs) > 0:

        # Pick box and add its index to the list
        i = ixs[0]
        ixs_pick.append(i)

        # Join this box to all other boxes with iou > threshold
        search_completed = False

        while not search_completed:
            # Compute IoU of the picked box with the rest
            iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
            # Identify boxes with IoU over the threshold. This
            # returns indicies into ixs[1:], so add 1 to get
            # indicies into ixs.
            join_ixs = np.where(iou > threshold)[0] + 1
            if len(join_ixs) > 0:
                new_box = np.array([min([boxes[i, 0], np.min(boxes[ixs[join_ixs], 0])]),
                                    max([boxes[i, 1], np.max(boxes[ixs[join_ixs], 1])]),
                                    min([boxes[i, 2], np.min(boxes[ixs[join_ixs], 2])]),
                                    max([boxes[i, 3], np.max(boxes[ixs[join_ixs], 3])])])
                new_mask = np.sum(np.stack([masks[:, :, i]] + [masks[:, :, j] for j in ixs[join_ixs]], axis = -1), axis = -1)
                boxes[i] = new_box
                masks[:, :, i] = new_mask
                scores[i] = (scores[i] + np.sum(scores[ixs[join_ixs]]))
                n_joins[i] = n_joins[i] + len(join_ixs)
                # Remove indicies of the overlapped boxes.
                ixs = np.delete(ixs, join_ixs)
            else:
                search_completed = True
                ixs = np.delete(ixs, 0)
    ixs_pick = np.array(ixs_pick)

    return ixs_pick, boxes[ixs_pick], scores[ixs_pick] / n_joins[ixs_pick], masks[:, :, ixs_pick], n_joins[ixs_pick]

