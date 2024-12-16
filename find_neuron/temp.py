import json
import torch

def make_mask_with_dict_and_save(
        path=r'/data/tigerbot/tigerbot_geely/test/fc/study/mask/xnli/en_2/500step/100000/neuron_dict_0.9996025.json',
        save_path=r"/data/tigerbot/tigerbot_geely/test/fc/study/mask/xnli/en_2/500step/100000/mask_cos.pt"):
    f = open(path, 'r', encoding='utf-8')
    neuron_dict = json.load(f)
    mask_dict = {}

    for layer_idx in range(32):
        print(layer_idx)
        for igo in ['in', 'out']:
            layer_name = str(layer_idx) + '_' + igo
            mask_dict[layer_name] = torch.zeros(11008 * 4096)
            if layer_name not in neuron_dict.keys():
                continue
            if igo == 'in' or igo == 'gate':
                for neuron in neuron_dict[layer_name]:
                    mask_dict[layer_name][neuron * 4096: (neuron + 1) * 4096] = 1
            else:
                for neuron in neuron_dict[layer_name]:
                    for i in range(4096):
                        index = i * 11008 + neuron
                        mask_dict[layer_name][index] = 1

    torch.save(mask_dict, save_path)

    return mask_dict


make_mask_with_dict_and_save()