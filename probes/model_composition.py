import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import gc
import einops
import torch
import argparse
import numpy as np
import pandas as pd
import datasets
import pickle
from utils import timestamp
from scipy.stats import spearmanr, pearsonr, linregress
from sklearn.metrics import r2_score
from save_activations import load_activation_probing_dataset_args
from probe_experiment import get_target_values_XU
from transformers import AutoTokenizer, AutoModelForCausalLM


import json

from transformer_lens import HookedTransformer

def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def get_top_neurons(learned_probes, model, k=100000, W_in=True):
    if W_in:
        W_norm = (model.W_in / model.W_in.norm(dim=1, keepdim=True)).swapaxes(1, 2)
    else:
        W_norm = (model.W_out / model.W_out.norm(dim=-1, keepdim=True))

    n_layers, d_mlp, d_model = W_norm.shape
    # print(W_norm.shape)

    W_comp = einops.einsum(W_norm, learned_probes.float(), 'l1 m d, l2 d-> l2 l1 m')

    top_neurons = W_comp.flatten().abs().argsort()

    print(len(top_neurons))
    _, top_layers, top_neurons = np.unravel_index(top_neurons[-k:], (learned_probes.shape[0], n_layers, d_mlp))
    print("top_layers: ")
    print(top_layers)
    print()
    print("top_neurons: ")
    print(top_neurons)
    print()
    print(_)

    neuron_dict = {}
    neuron_dict['top_layers'] = []
    neuron_dict['top_neurons'] = []
    neuron_dict['scores'] = []

    neuron_set = set()

    if W_in:
        neuron_weight = 'in'
    else:
        neuron_weight = 'out'


    for line in zip(_, top_layers, top_neurons):
        layer = line[0]
        neuron = line[1]
        neuron_name = str(layer) + '_' + str(neuron)
        neuron_set.add(neuron_name)

        score = W_comp[(line[0], line[1], line[2])]
        neuron_dict['top_layers'].append(str(line[1]))
        neuron_dict['top_neurons'].append(str(line[2]))
        print(score)
        print(float(score))
        neuron_dict['scores'].append(str(float(score)))
        # print(score)

    path = neuron_weight + '.json'
    with open(path, "w") as f:
        json.dump(neuron_dict, f)


    print('-------------------------------')





    return top_layers, top_neurons


def make_correlation_df(model, features, entity_activations, top_layers, top_neurons, W_in=True):
    neuron_corr = {}
    # print(len(top_layers))
    # print(len(top_neurons))
    for l, n in zip(top_layers, top_neurons):
        if W_in:
            neuron_probe = model.W_in[l, :, n]
        else:
            neuron_probe = model.W_out[l, n, :]

        for activation_layer, activations in entity_activations.items():
            neuron_probe_projection = activations @ neuron_probe

            # print(neuron_probe_projection)
            # print(features)
            # print(type(neuron_probe_projection))
            # print(type(features))
            # print(neuron_probe_projection.shape)
            # print(features.shape)
            # corr = spearmanr(neuron_probe_projection, features).correlation
            corr = pearsonr(neuron_probe_projection, features).statistic

            # R2
            # slope, intercept, r_value, p_value, corr = linregress(neuron_probe_projection, features)
            # corr = r2_score(features, neuron_probe_projection)


            neuron_corr[(l, n, activation_layer)] = corr

    corr_df = pd.DataFrame({'corr': pd.Series(neuron_corr)})
    corr_df.index.names = ['neuron_layer', 'neuron', 'activation_layer']
    corr_df = corr_df.reset_index()

    # print(corr_df.shape)

    return corr_df


def place_neuron_correlations(model, probe_result, place_df, entity_activations, top_k=50, start_layer=15):
    n_layers = model.config.num_hidden_layers
    lon_probes = torch.stack([
        torch.tensor(probe_result['probe_directions'][layer][:, 1])
        for layer in range(start_layer, n_layers)
    ])
    lon_probes = lon_probes / lon_probes.norm(dim=1, keepdim=True)

    lat_probes = torch.stack([
        torch.tensor(probe_result['probe_directions'][layer][:, 1])
        for layer in range(start_layer, n_layers)
    ])
    lat_probes = lat_probes / lat_probes.norm(dim=1, keepdim=True)

    top_neuron_dict = {
        ('lon', 'in'): get_top_neurons(lon_probes, model, k=top_k, W_in=True),
        ('lon', 'out'): get_top_neurons(lon_probes, model, k=top_k, W_in=False),
        ('lat', 'in'): get_top_neurons(lat_probes, model, k=top_k, W_in=True),
        ('lat', 'out'): get_top_neurons(lat_probes, model, k=top_k, W_in=False),
    }

    corr_dfs = []
    for (feature, neuron_weight), (top_layers, top_neurons, top_cos) in top_neuron_dict.items():
        feature_col = 'latitude' if feature == 'lat' else 'longitude'
        use_Win = neuron_weight == 'in'
        feature_values = place_df[feature_col].values

        corr_df = make_correlation_df(
            model, feature_values, entity_activations, top_layers, top_neurons, W_in=use_Win)
        corr_df['probe_cos'] = top_cos
        corr_df['feature'] = feature
        corr_df['neuron_weight'] = neuron_weight
        corr_dfs.append(corr_df)
    return pd.concat(corr_dfs)


def classifier_neuron_correlations_with_probes(model, probe_result, target, entity_activations, top_k=50,
                                               start_layer=15):
    n_layers = model.cfg.n_layers

    cls_probes = torch.stack([
        torch.tensor(probe_result['probe_directions'][layer][:, 1])
        for layer in range(start_layer, n_layers)
    ])

    cls_probes = cls_probes / cls_probes.norm(dim=1, keepdim=True)

    top_neuron_dict = {
        ('cls', 'in'): get_top_neurons(cls_probes, model, k=top_k, W_in=True),
        ('cls', 'out'): get_top_neurons(cls_probes, model, k=top_k, W_in=False)
    }

    # corr_dfs = []
    # for (feature, neuron_weight), (top_layers, top_neurons) in top_neuron_dict.items():
    #     if feature == 'cls':
    #         use_Win = neuron_weight == 'in'
    #         # feature_values = place_df[feature_col].values
    #
    #         corr_df = make_correlation_df(
    #             model, target, entity_activations, top_layers, top_neurons, W_in=use_Win)
    #         # print(corr_df)
    #
    #         # corr_df['probe_cos'] = top_cos
    #         corr_df['feature'] = pd.Series([feature] * corr_df.shape[0])
    #         corr_df['neuron_weight'] = pd.Series([neuron_weight] * corr_df.shape[0])
    #         corr_dfs.append(corr_df)
    #     else:
    #         print('feature error')
    # return pd.concat(corr_dfs)


def classifier_neuron_correlations_with_probes_input_pca_list(model, probe_result_list, target, entity_activations,
                                                              top_k=50,
                                                              start_layer=15):
    layer_key = 4096

    cls_probes = torch.stack([
        torch.tensor(probe_result['probe_directions'][layer_key][:, 1])
        for probe_result in probe_result_list
    ])

    print(cls_probes.shape)

    cls_probes = cls_probes / cls_probes.norm(dim=1, keepdim=True)

    top_neuron_dict = {
        ('cls', 'in'): get_top_neurons(cls_probes, model, k=top_k, W_in=True),
        ('cls', 'out'): get_top_neurons(cls_probes, model, k=top_k, W_in=False)
    }

    corr_dfs = []
    for (feature, neuron_weight), (top_layers, top_neurons) in top_neuron_dict.items():
        if feature == 'cls':
            use_Win = neuron_weight == 'in'
            # feature_values = place_df[feature_col].values

            corr_df = make_correlation_df(
                model, target, entity_activations, top_layers, top_neurons, W_in=use_Win)
            print(corr_df)

            # corr_df['probe_cos'] = top_cos
            corr_df['feature'] = pd.Series(feature)
            corr_df['neuron_weight'] = pd.Series(neuron_weight)
            corr_dfs.append(corr_df)
        else:
            print('feature error')
    return pd.concat(corr_dfs)


def RMSnorm(x, eps=1e-6):
    mean_sq = (x ** 2).mean(dim=1, keepdim=True)
    x = x / torch.sqrt(mean_sq + eps)
    return x


def pearson_correlation(matrix, target):
    n, d = matrix.size()
    target = target.view(-1, 1)  # reshape target to a column vector
    print(matrix)
    print(target)

    # Calculate the sums
    sum_x = matrix.sum(dim=0)
    sum_y = target.sum()
    sum_xy = (matrix * target).sum(dim=0)
    sum_xx = (matrix * matrix).sum(dim=0)
    sum_yy = (target * target).sum()

    # Compute the Pearson correlation for each column
    numerator = n * sum_xy - sum_x * sum_y
    denominator = torch.sqrt((n * sum_xx - sum_x ** 2)
                             * (n * sum_yy - sum_y ** 2))

    correlation = numerator / denominator
    print(correlation)


    return correlation

def r2_sklearn_correlation(matrix, target):
    # n, d = matrix.size()
    print(matrix)
    print(target)

    corr_list = []
    for m in matrix:
        corr = r2_score(target, m)
        corr_list.append(corr)

    print(corr_list)
    print('*************************')

    return corr_list


def spearman_correlation(matrix, target):
    n, d = matrix.size()
    target = target.view(-1, 1)  # reshape target to a column vector

    # Chunk neurons to reduce memory overhead
    chunk_size = 1024
    num_rows = matrix.size(0)
    matrix_ranks = torch.zeros_like(
        matrix, dtype=torch.float, device=matrix.device)
    for i in range(0, num_rows, chunk_size):
        chunk = matrix[:, i:i + chunk_size]
        rank_chunk = chunk.argsort(dim=0).argsort(dim=0).float() + 1.0
        matrix_ranks[:, i:i + chunk_size] = rank_chunk

    target_ranks = target.argsort(dim=0).argsort(
        dim=0).float() + 1.0  # convert to 1-indexed ranks

    # Calculate the sums
    sum_x = matrix_ranks.sum(dim=0)
    sum_y = target_ranks.sum()
    sum_xy = (matrix_ranks * target_ranks).sum(dim=0)
    sum_xx = (matrix_ranks * matrix_ranks).sum(dim=0)
    sum_yy = (target_ranks * target_ranks).sum()

    # Compute the Spearman correlation for each column
    numerator = n * sum_xy - sum_x * sum_y
    denominator = torch.sqrt((n * sum_xx - sum_x ** 2)
                             * (n * sum_yy - sum_y ** 2))

    correlation = numerator / denominator
    return correlation


def neuron_full_correlations(target_values, entity_activations, model, weight='W_in', use_spearman=True):
    target = torch.tensor(target_values).cuda()
    corrs = []
    for layer in range(model.config.num_hidden_layers - 1):
        acts = entity_activations[layer].cuda()
        acts = RMSnorm(acts)

        print(len(model.model.layers))  # 32

        if weight == 'W_in':
            weights = model.model.layers[layer + 1].mlp.up_proj.weight
        elif weight == 'W_gate':
            weights = model.model.layers[layer + 1].mlp.gate_proj.weight
        elif weight == 'W_out':
            weights = model.model.layers[layer].mlp.down_proj.weight.T
        else:
            raise ValueError(f'Invalid weight type: {weight}')

        weights = weights.cuda().to(torch.float32)
        neuron_acts = weights @ acts.T

        del acts
        del weights
        gc.collect()
        torch.cuda.empty_cache()

        # if use_spearman:
        #     corr = spearman_correlation(neuron_acts.T, target).detach().cpu()
        # else:
        #     corr = pearson_correlation(neuron_acts.T, target).detach().cpu()

        # R2
        # print(target)
        # print(neuron_acts.T)
        # print('*********************')
        #
        # corr = r2_score(target, neuron_acts.T)
        corr = r2_sklearn_correlation(neuron_acts.T, target)

        corrs.append(corr)

    full_corr = torch.stack(corrs, dim=0)
    return full_corr


def place_all_neuron_correlations(place_df, entity_activations, model, top_k=50):
    lat = place_df.latitude.values
    lon = place_df.longitude.values

    targets = {
        'lat': lat,
        'lon': lon,
    }
    if 'country' in place_df.columns:
        targets['abs_lat'] = np.abs(lat)
        targets['abs_lon'] = np.abs(lon)

    weights = ['W_in', 'W_gate', 'W_out']
    neuron_dfs = []
    for target_name, target_values in targets.items():
        for weight in weights:
            full_corr = neuron_full_correlations(
                target_values, entity_activations, model, weight=weight)
            top_ixs = full_corr.flatten().abs().argsort()[-top_k:]
            top_layers, top_neurons = np.unravel_index(
                top_ixs, full_corr.shape)
            df = pd.DataFrame({
                'feature': [target_name for _ in range(top_k)],
                'weight': [weight for _ in range(top_k)],
                # +1 because we skip the first layer
                'layer': top_layers + (1 if weight != 'W_out' else 0),
                'neuron': top_neurons,
                'corr': full_corr[top_layers, top_neurons],
                'abs_corr': full_corr[top_layers, top_neurons].abs()

            })
            neuron_dfs.append(df)
    return pd.concat(neuron_dfs)


def time_neuron_correlations(target, entity_activations, model, top_k=50):
    weights = ['W_in', 'W_gate', 'W_out']
    neuron_dfs = []
    for weight in weights:
        full_corr = neuron_full_correlations(
            target, entity_activations, model, weight=weight)
        top_ixs = full_corr.flatten().abs().argsort()[-top_k:]
        top_layers, top_neurons = np.unravel_index(
            top_ixs, full_corr.shape)
        df = pd.DataFrame({
            'feature': ['time' for _ in range(top_k)],
            'weight': [weight for _ in range(top_k)],
            # +1 because we skip the first layer
            'layer': top_layers + (1 if weight != 'W_out' else 0),
            'neuron': top_neurons,
            'corr': full_corr[top_layers, top_neurons],
            'abs_corr': full_corr[top_layers, top_neurons].abs()

        })
        neuron_dfs.append(df)
    return pd.concat(neuron_dfs)


def classifier_neuron_correlations(target, entity_activations, model, top_k=5000):
    # weights = ['W_in', 'W_gate', 'W_out']
    weights = ['W_in', 'W_out']

    neuron_dfs = []
    for weight in weights:
        full_corr = neuron_full_correlations(target, entity_activations, model, weight=weight, use_spearman=False)
        # print(full_corr[:1000])
        top_ixs = full_corr.flatten().abs().argsort()[-top_k:]
        top_layers, top_neurons = np.unravel_index(
            top_ixs, full_corr.shape)
        df = pd.DataFrame({
            'feature': ['classifier' for _ in range(top_k)],
            'weight': [weight for _ in range(top_k)],
            # +1 because we skip the first layer
            'layer': top_layers + (1 if weight != 'W_out' else 0),
            'neuron': top_neurons,
            'corr': full_corr[top_layers, top_neurons],
            'abs_corr': full_corr[top_layers, top_neurons].abs()

        })
        neuron_dfs.append(df)
    return pd.concat(neuron_dfs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', default='Llama-2-7b-chat-hf',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--entity_type',
        help='Name of feature collection (should be dir under processed_datasets/)')
    parser.add_argument(
        '--feature_name', type=str, default='coords',
        help='Name of feature to probe, must be in FEATURE_PROMPT_MAPPINGS')
    parser.add_argument(
        '--prompt_name', type=str,
        help='Name of prompt to use for probing, must key in <ENTITY>_PROMPTS')
    parser.add_argument(
        '--activation_aggregation', default='mean',
        help='Average activations across all tokens in a sequence')
    parser.add_argument(
        '--corr_with_probes', default=False, action='store_true',
        help='Average activations across all tokens in a sequence')
    parser.add_argument(
        '--is_pca', default=False, action='store_true',
        help='Average activations across all tokens in a sequence')

    args = parser.parse_args()
    torch.set_grad_enabled(False)

    if args.model == 'Llama-2-7b-chat-hf':
        model_path = r'/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf'
    elif args.model == 'xnli_en_sft_all_layers_500steps':
        model_path = r'/mnt/nfs/algo/intern/haoyunx11/models/sft/xnli/en/checkpoint-500'
    elif args.model == 'xnli_en_sft_layer4_2500steps':
        model_path = r'/mnt/nfs/algo/intern/haoyunx11/models/sft/xnli/en_sft_layer4/checkpoint-2500'
    elif args.model == 'xnli_en_sft_layer30_2500steps':
        model_path = r'/mnt/nfs/algo/intern/haoyunx11/models/sft/xnli/en_sft_layer30/checkpoint-2500'
    elif args.model == 'neuron_4_328_in_out_progressive_500steps':
        model_path = r'/mnt/nfs/algo/intern/haoyunx11/models/sft/xnli/train_neuron/neuron_4_328_in_out_progressive/checkpoint-500'
    elif args.model == 'neuron_4_328_in_out_progressive_2000steps':
        model_path = r'/mnt/nfs/algo/intern/haoyunx11/models/sft/xnli/train_neuron/neuron_4_328_in_out_progressive/checkpoint-2000'

    if not args.corr_with_probes:
        # model self correlaton
        print('model self correlaton')

        model = AutoModelForCausalLM.from_pretrained(model_path)

        n_layers = model.config.num_hidden_layers
        entity_activations = {l: load_activation_probing_dataset_args(args, args.prompt_name, l).dequantize()
                              for l in range(n_layers)}

        entity_type_split = args.entity_type.split('.')

        data = datasets.load_from_disk(
            r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/sn/data/' + 'xnli/' + entity_type_split[
                1] + '/' + args.prompt_name)
        target = get_target_values_XU(data, args.entity_type)
        target = np.array(target)

        # print(target)

        print(timestamp(), f'running neuron composition on {args.model} {args.entity_type}')

        neuron_df = classifier_neuron_correlations(target, entity_activations, model)

        save_path = os.path.join(r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results',
                                 'top_neurons', args.model, entity_type_split[1])
        os.makedirs(save_path, exist_ok=True)
        neuron_df.to_csv(
            os.path.join(save_path, 'self_correlaton' + '_' + args.entity_type + '_' + args.prompt_name + '_pearson.csv'),
            index=False)

    else:
        # correlation with probes direction
        print('correlation with probes direction')

        if not args.is_pca:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            hf_model = AutoModelForCausalLM.from_pretrained(model_path)
            model = HookedTransformer.from_pretrained(r'Llama-2-7b-chat-hf', hf_model=hf_model, device="cpu",
                                                      fold_ln=False,
                                                      center_writing_weights=False, center_unembed=True,
                                                      tokenizer=tokenizer)

            n_layers = model.cfg.n_layers
            entity_activations = {l: load_activation_probing_dataset_args(args, args.prompt_name, l).dequantize()
                                  for l in range(n_layers)}

            entity_type_split = args.entity_type.split('.')

            data = datasets.load_from_disk(
                r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/sn/data/xnli/' + entity_type_split[
                    1] + '/' + args.prompt_name)
            target = get_target_values_XU(data, args.entity_type)
            target = np.array(target)

            entity_type_split = args.entity_type.split('.')

            probe_result = pickle.load(open(
                r"/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/Llama-2-7b-chat-hf/" +
                entity_type_split[0] + "/" + entity_type_split[
                    1] + "/" + args.entity_type + ".mean." + args.prompt_name + ".p",
                'rb'))

            corr_df = classifier_neuron_correlations_with_probes(model, probe_result, target, entity_activations,
                                                                 top_k=150000,
                                                                 start_layer=0)

            corr_df['abs_corr'] = np.abs(corr_df['corr'].values)
            corr_df.sort_values('abs_corr', ascending=False).groupby(['neuron_layer', 'neuron']).head(1)

            save_path = os.path.join(r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/final',
                                     'top_neurons', args.model, entity_type_split[1])
            os.makedirs(save_path, exist_ok=True)
            corr_df.to_csv(os.path.join(save_path, args.entity_type + '_' + args.prompt_name + '_pearsonr.csv'), index=False)

        else:
            print('pca activation')
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            hf_model = AutoModelForCausalLM.from_pretrained(model_path)
            model = HookedTransformer.from_pretrained(r'Llama-2-7b-chat-hf', hf_model=hf_model, device="cpu",
                                                      fold_ln=False,
                                                      center_writing_weights=False, center_unembed=True,
                                                      tokenizer=tokenizer)

            n_layers = model.cfg.n_layers
            entity_activations = {l: load_activation_probing_dataset_args(args, args.prompt_name, l).dequantize()
                                  for l in range(n_layers)}

            data = datasets.load_from_disk(r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/sn/data/xnli/en/simple_prompt')
            target = get_target_values_XU(data, args.entity_type)
            target = np.array(target)

            # temp 写法
            probe_result_list = []
            probe_result = pickle.load(open(
                r"/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/Llama-2-7b-chat-hf/xnli/en/xnli.en.mean.simple_prompt_pca_layer11.p",
                'rb'))
            probe_result_list.append(probe_result)

            probe_result = pickle.load(open(
                r"/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/Llama-2-7b-chat-hf/xnli/en/xnli.en.mean.simple_prompt_pca_layer12.p",
                'rb'))
            probe_result_list.append(probe_result)

            corr_df = classifier_neuron_correlations_with_probes_input_pca_list(model, probe_result_list, target,
                                                                                entity_activations, top_k=50,
                                                                                start_layer=0)

            corr_df['abs_corr'] = np.abs(corr_df['corr'].values)
            corr_df.sort_values('abs_corr', ascending=False).groupby(['neuron_layer', 'neuron']).head(1)

            save_path = os.path.join(r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results',
                                     'top_neurons', args.model)
            os.makedirs(save_path, exist_ok=True)
            corr_df.to_csv(os.path.join(save_path, args.entity_type), index=False)

"""

# corr with probe R2

python -u model_composition.py \
    --model Llama-2-7b-chat-hf \
    --entity_type xnli.bg \
    --prompt_name standard_prompt \
    --corr_with_probes >model_composition_xnli_org.bg_standard_prompt_r2_sklearn.out 2>&1 &


# model self corr R2

python -u model_composition.py \
    --model Llama-2-7b-chat-hf \
    --entity_type xnli.en \
    --prompt_name standard_prompt >model_self_corr_xnli_org.en_standard_prompt_pearson.out 2>&1 &
    
python -u model_composition.py \
    --model xnli_en_sft_all_layers_500steps \
    --entity_type xnli_en_sft_all_layers_500steps.en \
    --prompt_name standard_prompt >model_self_corr_xnli_xnli_en_sft_all_layers_500steps.en_standard_prompt_r2_sklearn.out 2>&1 &


"""

