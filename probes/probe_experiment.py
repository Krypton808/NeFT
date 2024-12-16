import os
import sys
import datasets
import random

sys.path.append(r'/data5/haoyun.xu/study/MI/world-models')

import pickle
import argparse
import tqdm

import numpy as np
import pandas as pd

from utils import timestamp, MODEL_N_LAYERS, get_model_family
# from make_prompt_datasets import ENTITY_PROMPTS
from save_activations import load_activation_probing_dataset, load_activation_probing_dataset_args
# from probes import baseline
# from probes import rank
from probes.evaluation import *
from feature_datasets import common
from sklearn.linear_model import Ridge, RidgeCV, RidgeClassifier, RidgeClassifierCV
from sklearn import metrics
from scipy import stats

import warnings

warnings.filterwarnings(
    'ignore', category=UserWarning,
    message='TypedStorage is deprecated')


def save_probe_results(args, probe_results, prompt, pca=False):
    save_path = os.path.join(
        os.getenv('RESULTS_DIR', 'results'),
        args.experiment_name,
        args.model,
        args.entity_type,
        args.feature_name,
        'probes' if not pca else 'pca_probes'
    )
    os.makedirs(save_path, exist_ok=True)

    probe_metadata = [
        args.normalization_type,
        args.label_processing,
        args.activation_aggregation,
        prompt
    ]
    probe_name = '.'.join(probe_metadata) + '.p'

    pickle.dump(
        probe_results,
        open(os.path.join(save_path, probe_name), 'wb')
    )


def save_probe_results_temp(args, probe_results):
    entity_type_split = args.entity_type.split('.')

    pickle.dump(
        probe_results,
        open(
            r"/data/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/Llama-2-7b-chat-hf/" +
            entity_type_split[0] + "/" + entity_type_split[
                1] + "/" + args.entity_type + ".mean." + args.prompt_name + ".p",
            'wb')
    )


def load_probe_results(
        experiment_name, model, entity_type, feature_name, prompt,
        experiment_type='lsq_train',
        normalization_type='none',
        label_processing='none',
        activation_aggregation='last',
        pca=False):
    save_path = os.path.join(
        os.getenv('RESULTS_DIR', 'results'),
        experiment_name,
        model,
        entity_type,
        feature_name,
        'probes' if not pca else 'pca_probes'
    )
    os.makedirs(save_path, exist_ok=True)

    probe_metadata = [
        experiment_type,
        normalization_type,
        label_processing,
        activation_aggregation,
        prompt
    ]
    probe_name = '.'.join(probe_metadata) + '.p'

    probe_results = pickle.load(
        open(os.path.join(save_path, probe_name), 'rb'))

    return probe_results


def place_probe_experiment(activations, target, is_test, probe=None, is_lat_lon=True):
    train_activations = activations[~is_test]
    train_target = target[~is_test]

    test_activations = activations[is_test]
    test_target = target[is_test]

    norm_train_target = (
                                train_target - train_target.mean(axis=0)) / train_target.std(axis=0)

    if probe is None:
        probe = Ridge(alpha=activations.shape[1])

    print(train_activations.shape)
    print(test_activations.shape)
    # print(norm_train_target.shape)
    # print(train_target)

    probe.fit(train_activations, norm_train_target)

    train_pred = probe.predict(train_activations)
    test_pred = probe.predict(test_activations)

    train_pred_unnorm = train_pred * \
                        train_target.std(axis=0) + train_target.mean(axis=0)
    test_pred_unnorm = test_pred * \
                       train_target.std(axis=0) + train_target.mean(axis=0)

    projection = probe.predict(activations) * \
                 train_target.std(axis=0) + train_target.mean(axis=0)

    train_scores = score_place_probe(
        train_target, train_pred_unnorm, use_haversine=is_lat_lon)
    test_scores = score_place_probe(
        test_target, test_pred_unnorm, use_haversine=is_lat_lon)

    scores = {
        **{('train', k): v for k, v in train_scores.items()},
        **{('test', k): v for k, v in test_scores.items()},
    }

    error_matrix = compute_proximity_error_matrix(
        target, projection, pairwise_haversine_distance)

    train_error, test_error, combined_error = proximity_scores(
        error_matrix, is_test)
    scores['train', 'prox_error'] = train_error.mean()
    scores['test', 'prox_error'] = test_error.mean()

    projection_df = pd.DataFrame({
        'x': projection[:, 0],
        'y': projection[:, 1],
        'is_test': is_test,
        'x_error': projection[:, 0] - target[:, 0],
        'y_error': projection[:, 1] - target[:, 1],
        'prox_error': combined_error,
    })
    return probe, scores, projection_df


def time_probe_experiment(activations, target, is_test, probe=None):
    train_activations = activations[~is_test]
    train_target = target[~is_test]

    test_activations = activations[is_test]
    test_target = target[is_test]

    norm_train_target = (train_target - train_target.mean()
                         ) / train_target.std()

    if probe is None:
        probe = Ridge(alpha=activations.shape[1])

    probe.fit(train_activations, norm_train_target)

    train_pred = probe.predict(train_activations)
    test_pred = probe.predict(test_activations)

    train_pred_unnorm = train_pred * train_target.std() + train_target.mean()
    test_pred_unnorm = test_pred * train_target.std() + train_target.mean()

    projection = probe.predict(activations) * \
                 train_target.std() + train_target.mean()

    train_scores = score_time_probe(train_target, train_pred_unnorm)
    test_scores = score_time_probe(test_target, test_pred_unnorm)
    scores = {
        **{('train', k): v for k, v in train_scores.items()},
        **{('test', k): v for k, v in test_scores.items()},
    }

    error_matrix = compute_proximity_error_matrix(
        target, projection, pairwise_abs_distance_fn)

    train_error, test_error, combined_error = proximity_scores(
        error_matrix, is_test)
    scores['train', 'prox_error'] = train_error.mean()
    scores['test', 'prox_error'] = test_error.mean()

    projection_df = pd.DataFrame({
        'projection': projection,
        'is_test': is_test,
        'error': projection - target,
        'prox_error': combined_error,
    })

    return probe, scores, projection_df


def classifier_probe_experiment(activations, target, is_test, probe):
    # classifier 计算acc，不用做norm
    train_activations = activations[~is_test]
    train_target = target[~is_test]

    test_activations = activations[is_test]
    test_target = target[is_test]

    probe.fit(train_activations, train_target)

    train_pred = probe.predict(train_activations)
    test_pred = probe.predict(test_activations)

    train_scores = score_classifier_probe(train_target, train_pred)
    test_scores = score_classifier_probe(test_target, test_pred)

    return probe, train_scores, test_scores


def train_test_split(input, random_list):
    train_list = []
    test_list = []
    for idx, i in enumerate(input):
        if idx in random_list:
            test_list.append(i)
            continue

        train_list.append(i)

    return train_list, test_list


def get_target_values(entity_df, feature_name):
    if feature_name == 'coords':
        target = entity_df[['longitude', 'latitude']].values

    elif feature_name.endswith('date') or feature_name.endswith('year'):
        if feature_name == 'death_year':
            target = entity_df[feature_name].values
        else:
            NS_PER_YEAR = 1e9 * 60 * 60 * 24 * 365.25
            target = pd.to_datetime(entity_df[feature_name]).values
            target = target.astype(np.int64) / NS_PER_YEAR
    else:
        raise ValueError(f'Unrecognized feature name: {feature_name}')
    return target


def get_target_values_XU(data, entity_type):
    if 'xnli' in entity_type or 'neuron' in entity_type:
        target = data['label']

    else:
        raise ValueError(f'Unrecognized feature name: {entity_type}')
    return target


MODEL_ALPHA = {
    'Llama-2-7b-hf': 5000,
    'Llama-2-7b-chat-hf': 5000,
    'Llama-2-13b-hf': 10000,
    'Llama-2-70b-hf': 20000,
}

RIDGE_ALPHAS = {
    'Llama-2-7b-hf': np.logspace(0.8, 4.1, 12),
    'Llama-2-7b-chat-hf': np.logspace(0.8, 4.1, 12),
    'Llama-2-13b-hf': np.logspace(0.8, 4.3, 12),
    'Llama-2-70b-hf': np.logspace(0.8, 4.5, 12),
}


def main_probe_experiment(args):
    random.seed(1234)
    n_layers = MODEL_N_LAYERS[args.model]

    # entity_df = pd.read_csv(r'/data5/haoyun.xu/study/MI/world-models/data/entity_datasets/world_place.csv')
    # is_test = entity_df.is_test.values

    entity_type_split = args.entity_type.split('.')

    data = datasets.load_from_disk(
        r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/sn/data/xnli/' + entity_type_split[1] + '/' + args.prompt_name)

    data_length = len(data)
    random_num = random.sample(range(data_length), 1000)
    is_test = []
    for i in range(data_length):
        if i in random_num:
            is_test.append(True)
        else:
            is_test.append(False)

    is_test = np.array(is_test)

    print(timestamp(),
          f'running probe on {args.model} {args.entity_type}.{args.prompt_name}')

    results = {
        'train_score': {},
        'test_score': {},
        'probe_directions': {},
        'probe_biases': {},
        'probe_alphas': {},
    }

    # target = get_target_values(entity_df, args.feature_name)
    # print(target)

    target = get_target_values_XU(data, args.entity_type)
    target = np.array(target)
    # train_tgt, test_tgt = train_test_split(target, random_num)

    for layer in tqdm.tqdm(range(n_layers)):
        # load data
        activations = load_activation_probing_dataset_args(args, args.prompt_name, layer).dequantize()
        # train_act, test_act = train_test_split(activations, random_num)

        if activations.isnan().any():
            print(timestamp(), 'WARNING: nan activations, skipping layer', layer)
            continue

        # probe = RidgeCV(alphas=RIDGE_ALPHAS[args.model], store_cv_values=True)
        probe = RidgeClassifierCV(alphas=RIDGE_ALPHAS[args.model], store_cv_values=True)
        probe, train_score, test_score = classifier_probe_experiment(activations, target, is_test, probe)
        probe_direction = probe.coef_.T.astype(np.float16)
        probe_alphas = probe.cv_values_.mean(axis=0)

        results['train_score'][layer] = train_score
        results['test_score'][layer] = test_score
        results['probe_directions'][layer] = probe_direction
        results['probe_biases'][layer] = probe.intercept_
        results['probe_alphas'][layer] = probe_alphas

    save_probe_results_temp(args, results)


def main_probe_experiment_old(args):
    n_layers = MODEL_N_LAYERS[args.model]

    entity_df = common.load_entity_data(args.entity_type)
    is_test = entity_df.is_test.values

    print(timestamp(),
          f'running probe on {args.model} {args.experiment_type}.{args.feature_name}.{args.prompt_name}')

    results = {
        'scores': {},
        'projections': {},
        'probe_directions': {},
        'probe_biases': {},
        'probe_alphas': {},
    }
    for layer in tqdm.tqdm(range(n_layers)):
        # load data
        activations = load_activation_probing_dataset_args(
            args, args.prompt_name, layer).dequantize()

        if activations.isnan().any():
            print(timestamp(), 'WARNING: nan activations, skipping layer', layer)
            continue

        target = get_target_values(entity_df, args.feature_name)

        probe = RidgeCV(alphas=RIDGE_ALPHAS[args.model], store_cv_values=True)

        is_place = args.entity_type.endswith('place')

        if is_place:
            probe, scores, projection = place_probe_experiment(
                activations, target, is_test, probe=probe)

        else:
            print(activations)
            print('***********************')
            print(target)
            probe, scores, projection = time_probe_experiment(
                activations, target, is_test, probe=probe)
            break

        probe_direction = probe.coef_.T.astype(np.float16)
        probe_alphas = probe.cv_values_.mean(axis=(0, 1) if is_place else 0)

        results['scores'][layer] = scores
        results['projections'][layer] = projection
        results['probe_directions'][layer] = probe_direction
        results['probe_biases'][layer] = probe.intercept_
        results['probe_alphas'][layer] = probe_alphas

    save_probe_results(args, results, args.prompt_name)


def pca_probe_experiment(args):
    random.seed(1234)

    MODEL_LAYER = {
        'Llama-2-7b-chat-hf': 11,
        'Llama-2-7b-hf': 20,
        'Llama-2-13b-hf': 24,
        'Llama-2-70b-hf': 48,
    }
    layer = MODEL_LAYER[args.model]

    data = datasets.load_from_disk(r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/sn/data/xnli/en/simple_prompt')
    data_length = len(data)
    random_num = random.sample(range(data_length), 1000)
    is_test = []
    for i in range(data_length):
        if i in random_num:
            is_test.append(True)
        else:
            is_test.append(False)

    is_test = np.array(is_test)

    activations = load_activation_probing_dataset_args(
        args, args.prompt_name, layer).dequantize()

    PCA_DIMS = [2, 4, 6, 8, 10, 15, 20, 30,
                40, 50, 75, 100, 512, 1024, 2048, activations.shape[1]]

    print(timestamp(),
          f'running probe on {args.model} {args.experiment_type}.{args.feature_name}.{args.prompt_name}')

    U, S, V = torch.pca_lowrank(activations, q=PCA_DIMS[-2])

    results = {
        'train_score': {},
        'test_score': {},
        'probe_directions': {},
        'probe_biases': {},
        'probe_alphas': {},
    }

    for pca_dim in tqdm.tqdm(PCA_DIMS):
        # load data

        if activations.isnan().any():
            print(timestamp(), 'WARNING: nan activations, skipping layer', pca_dim)
            continue

        if pca_dim != PCA_DIMS[-1]:
            pca_activations = activations @ V[:, :pca_dim]
        else:
            pca_activations = activations

        target = get_target_values_XU(data, args.entity_type)
        target = np.array(target)

        # probe = RidgeCV(alphas=RIDGE_ALPHAS[args.model], store_cv_values=True)
        probe = RidgeClassifierCV(alphas=RIDGE_ALPHAS[args.model], store_cv_values=True)

        probe, train_score, test_score = classifier_probe_experiment(pca_activations, target, is_test, probe)

        probe_direction = probe.coef_.T.astype(np.float16)
        probe_alphas = probe.cv_values_.mean(axis=0)

        results['train_score'][pca_dim] = train_score
        results['test_score'][pca_dim] = test_score
        results['probe_directions'][pca_dim] = probe_direction
        results['probe_biases'][pca_dim] = probe.intercept_
        results['probe_alphas'][pca_dim] = probe_alphas

    save_probe_results_temp(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment params
    parser.add_argument(
        '--experiment_name', type=str, help='Name of experiment for save dir')
    parser.add_argument(
        '--experiment_type', type=str, default='lsq_train',
        help='Type of experiment: spearman_train, kendall_train, lsq_train')
    parser.add_argument(
        '--model', default='pythia-70m',
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
        '--normalization_type', type=str, default='none',
        help='Type of normalization to apply to activations before training')
    parser.add_argument(
        '--label_processing', type=str, default='none',
        help='Type of weighting to apply to labels before training')
    parser.add_argument(
        '--activation_aggregation', default='mean',
        help='Average activations across all tokens in a sequence')
    parser.add_argument(
        '--pca', action='store_true')
    # parser.add_argument(
    #     '--probe_device', default='cpu',
    #     help='Device to run probe training on (only relevant for kendall tau)')

    # TODO: potentially add more probe type experiments

    args = parser.parse_args()

    if not args.pca:
        main_probe_experiment(args)
    else:
        pca_probe_experiment(args)
