import torch

def dic_to_tensor(dic, keys=None):
    """ convert tactile array dic to (stacked) tensor """
    out = []
    if keys is None:
        for key, val in dic.items():
            out.append(torch.tensor(val)) if not isinstance(val, torch.Tensor) else out.append(val)
        keys = list(dic.keys())
    else:
        for key in keys:
            val = dic[key]
            out.append(torch.tensor(val)) if not isinstance(val, torch.Tensor) else out.append(val)
    out = torch.stack(out, dim=0)
    return keys, out

def tensor_to_dic(keys, tensor):
    """ convert (stacked) tensor to dic """
    dic = {keys[i]:tensor[i] for i in range(len(keys))}
    return dic

def rearrange_dic(dic, keys):
    """ rearrange the dic based on keys """
    return {key:dic[key] for key in keys}
