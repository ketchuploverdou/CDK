import os
import sys
import time
import warnings
from collections import defaultdict
import pandas as pd

import paddle
import numpy as np
import paddle.distributed as dist
from paddle.optimizer.lr import StepDecay

from reader import read_trigraph
from dataset import create_dataloaders
from model import KGEModel
from loss_func import LossFunction
from utils import set_seed, set_logger, print_log
from utils import evaluate
from config import prepare_config


def predict(model, loader, save_path):
    model.eval()
    top_tens = []
    pred = np.zeros(shape=[1, 12], dtype=np.int32)
    with paddle.no_grad():
        for h, r, t in loader:
            score = model.predict(h, r, mode='head')
            rank = paddle.argsort(score, axis=1, descending=True)
            tmp = np.concatenate([h.numpy()[:, None], r.numpy()[:, None], rank[:, :10].numpy()], axis=1)
            pred = np.concatenate([pred, tmp], axis=0)
    pred = pred[1:, :]
    print(pred.shape)
    columns = ['head', 'relation', 't_0', 't_1', 't_2', 't_3', 't_4', 't_5', 't_6', 't_7', 't_8', 't_9']
    data_dict = {columns[i]: pred[:, i] for i in range(12)}
    df = pd.DataFrame(data_dict)
    df.to_csv(os.path.join(save_path, 'top_10.csv'), index=False)


def main():
    """Main function for shallow knowledge embedding methods.
    """
    args = prepare_config()
    set_seed(args.seed)
    set_logger(args)

    trigraph = read_trigraph(args.data_path, args.data_name, args.use_dict,
                             args.kv_mode)
    if args.valid_percent < 1:
        trigraph.sampled_subgraph(args.valid_percent, dataset='valid')

    use_filter_set = args.filter_sample or args.filter_eval or args.weighted_loss
    if use_filter_set:
        filter_dict = {
            'head': trigraph.true_heads_for_tail_rel,
            'tail': trigraph.true_tails_for_head_rel
        }
    else:
        filter_dict = None

    if dist.get_world_size() > 1:
        dist.init_parallel_env()

    model = KGEModel(args.model_name, trigraph, args)

    train_loader, valid_loader, test_loader = create_dataloaders(
        trigraph,
        args,
        filter_dict=filter_dict if use_filter_set else None,
        shared_ent_path=model.shared_ent_path if args.mix_cpu_gpu else None)
    if args.init_from_ckpt:
        state_dict = paddle.load(
            os.path.join(args.init_from_ckpt, 'params.pdparams'))
        model.set_dict(state_dict)
    predict(model, test_loader, args.save_path)


if __name__ == '__main__':
    main()
