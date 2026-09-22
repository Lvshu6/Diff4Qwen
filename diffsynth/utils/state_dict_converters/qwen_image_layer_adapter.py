def QwenImageLayerAdapterStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("layer_adapter."):
            state_dict_[name] = state_dict[name]
    return state_dict_