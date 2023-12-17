import os
import random
import logging
import argparse
import torch
import numpy as np
import pandas as pd

from config import ConfigRegression
from data.load_data import MMDataLoader
from lstm_models.model import GCD_CMR
from lstm_models.train_test import Train_Test
from utils.functions import prep_optimizer
import pynvml

import gc
import shutil

import datetime



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_once(args):
    args.model_save_dir = os.path.join(args.res_save_dir, "model")
    os.makedirs(args.model_save_dir, exist_ok=True)
    align_mode = "aligned" if args.aligned else "unaligned"
    args.model_save_path = os.path.join(
        args.model_save_dir, f'{args.modelName}-{align_mode}-{args.datasetName}.pth')

    if not args.gpu_ids and torch.cuda.is_available():
        pynvml.nvmlInit()
        dst_gpu_id, min_mem_used = 0, 1e16
        for g_id in [0, 1]:
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        print(f'Find gpu: {dst_gpu_id}, use memory: {min_mem_used}!')
        logger.info(
            f'Find gpu: {dst_gpu_id}, with memory: {min_mem_used} left!')
        args.gpu_ids.append(dst_gpu_id)

    using_cuda = bool(args.gpu_ids) and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu_ids[0]}' if using_cuda else 'cpu')
    args.device = device
    dataloader = MMDataLoader(args)
    model = GCD_CMR(args).to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f'The model has {count_parameters(model)} trainable parameters')

    optimizer, scheduler, model = prep_optimizer(args, model)
    trainer = Train_Test(args, model, optimizer, scheduler, dataloader['train'], dataloader['valid'], dataloader['test'], logger)
    trainer.do_train()

    assert os.path.exists(args.model_save_path)
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(args.device)
    val_results = trainer.do_test(model, mode="VAL")
    results = trainer.do_test(model, mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results

def run(args):
    init_args = args
    model_results = []
    seeds = args.seeds

    for i, seed in enumerate(seeds):
        args = init_args
        if args.train_mode == "regression":
            config = ConfigRegression(args)
        else:
            config = ConfigClassification(args)
        args = config.get_config()
        setup_seed(seed)
        args.seed = seed
        logger.info(f'Start running {args.modelName}...')
        logger.info(args)

        args.cur_time = i + 1
        test_results = run_once(args)
        model_results.append(test_results)

    last_results = list(model_results[0].keys())
    logger.info('Results:')
    logger.info('Criterion: Mean +- Std')
    for idx in last_results:
        values = [r[idx] for r in model_results]
        mean_value = np.mean(values) * 100
        std_value = np.std(values) * 100
        logger.info(f'{idx}: {round(mean_value, 2)}, {round(std_value, 2)}')

    best_result_index = np.argmax([r[last_results[2]] for r in model_results])
    best_seed = best_result_index + 1111
    logger.info(f'best result seed is {best_seed}')
    logger.info('the best results are:')
    for i, idx in enumerate(last_results):
        logger.info(f'{idx}: {model_results[best_result_index][idx]}')

    logger.info('Results have been added')
    list_name = list(model_results[0].keys())
    index1 = [5, 2, 0, 3, 1, 6, 7, 4, 8]
    list_name = [list_name[i] for i in index1]
    list_max = list_name

    align_mode = "align" if args.aligned else "no_align"
    csv_path = os.path.join(args.csv_path, f'{args.datasetName}-{align_mode}.csv')

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["Task"] + list_name)

    current_time = datetime.datetime.now().strftime('%Y-%m-%d')
    res = [f'{args.task_name}-average-{current_time}']
    for c in list_name:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values) * 100, 2)
        std = round(np.std(values) * 100, 2)
        res.append(mean)

    df.loc[len(df)] = res
    df.to_csv(csv_path, index=None)
    logger.info(f'Results are added to {csv_path}...')


def set_log(args):
    log_file_path = os.path.join(args.res_save_dir, 'log.log')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers:
        logger.removeHandler(ph)

    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)

    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)

    return logger



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", type=bool, default=True, help="whether to train aligned or unaligned dataset")
    parser.add_argument('--datasetName', type=str, default='mosi', help='support mosi/mosei')

    parser.add_argument('--circle_time', type=int, default=3, required=False, help="time for decomposition")
    parser.add_argument('--epochs', type=int, default=30, required=False, help="training epochs")
    parser.add_argument('--diff_weight', type=float, default=0.1, required=False, help="weight of diff loss")
    parser.add_argument('--spc_weight', type=float, default=0.05, required=False, help="weight of spc loss")
    parser.add_argument('--gpu_ids', type=list, default=[4], help='indicates the GPUs to be used. If none, the most free GPU will be used!')
    parser.add_argument('--seeds', type=list, default=[1115], required=False)
    parser.add_argument("--batch_size", default=64, type=int, required=False, help="Batch size for training.")
    parser.add_argument("--bidirectional", default=False, type=bool, required=False, help="whether to use bidirectional")

    parser.add_argument('--is_tune', type=bool, default=False, help='tune parameters?')
    parser.add_argument('--train_mode', type=str, default="regression", help='regression / classification')
    parser.add_argument('--modelName', type=str, default='gcd_cmr', help='name of the training model')

    parser.add_argument('--num_workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--task_name', type=str, default='test', help='task name')
    parser.add_argument('--res_save_dir', type=str, default='results', help='path to save results.')

    args = parser.parse_args()
    args.csv_path = args.res_save_dir
    
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    aligned_mode = 'aligned' if args.aligned else 'unaligned'
    task_name = args.task_name  + args.datasetName + '_' + aligned_mode + '_' + current_time
    args.res_save_dir = os.path.join(args.res_save_dir, task_name)
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)

    return args

if __name__ == '__main__':
    args = parse_args()
    logger = set_log(args)
    run(args)
