from collections import OrderedDict

def remove_module(ckpt):
    state_dict = OrderedDict()
    for key, value in ckpt.items():
        name = key.replace('module.', '')
        state_dict[name] = value
    return state_dict