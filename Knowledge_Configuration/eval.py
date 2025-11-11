import os
import sys
import time
import warnings
from collections import defaultdict

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


def main():
    """Main function for shallow knowledge embedding methods.
    """
    args = prepare_config()
    set_seed(args.seed)
    set_logger(args)

    trigraph = read_trigraph(args.data_path, args.data_name, args.use_dict, 'kv')
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

    model = KGEModel(args.model_name, trigraph, args)

    _, _, test_loader = create_dataloaders(
        trigraph,
        args,
        filter_dict=filter_dict if use_filter_set else None,
        shared_ent_path=model.shared_ent_path if args.mix_cpu_gpu else None)
    if args.init_from_ckpt:
        state_dict = paddle.load(
            os.path.join(args.init_from_ckpt, 'params.pdparams'))
        model.set_dict(state_dict)
    evaluate(
        model,
        test_loader,
        'test',
        filter_dict if args.filter_eval else None,
        args.save_path,
        data_mode=args.data_name)


if __name__ == '__main__':
    main()
