import os

# Suppress TF/XLA noisy logs if something imports tensorflow indirectly
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pickle
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import get_logger, ddp_setup
from trainer import Trainer
from models import *
from meta_models import *
from CIFAR.data_helper_cifar import prepare_data


parser = argparse.ArgumentParser(description='EMLC Training Framework')

# General and paths
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'])  # Required
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--data_seed', type=int, default=0)
parser.add_argument('--runid', default='clothing1m_run_best', type=str)
parser.add_argument('--data_path', default='data/', type=str, help='Root for the datasets.')
parser.add_argument('--logdir', type=str, default='runs', help='Log folder.')

# SSL path optional now (we will train from scratch regardless)
parser.add_argument('--ssl_path', type=str, default=None, help='SSL pretrained model path (optional).')

# Optional: disable tensorboard to keep logs clean
parser.add_argument('--no_tb', action='store_true', help='Disable TensorBoard logging (clean console output).')

# Training
parser.add_argument('--epochs', '-e', type=int, default=15, help='Number of epochs to train.')
parser.add_argument('--every', default=10, type=int, help='Eval interval')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--test_bs', default=100, type=int, help='batch size')
parser.add_argument('--gold_bs', type=int, default=32)
parser.add_argument('--embedding_dim', type=int, default=512, help='Feature extractor output dim')
parser.add_argument('--label_embedding_dim', type=int, default=128, help='Label embedding dim')
parser.add_argument('--mlp_hidden_dim', type=int, default=128, help='MLP hidden layer units')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optimizer')
parser.add_argument('--main_lr', default=2e-2, type=float, help='lr for main net')
parser.add_argument('--meta_lr', default=2e-2, type=float, help='lr for meta net')
parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--gradient_steps', default=5, type=int, help='Number of look-ahead gradient steps for meta-gradient')
parser.add_argument('--sched_milestones', default='20', type=str, help='Milestones in which LR is decreased')
parser.add_argument('--sched_gamma', default=0.1, type=float, help='Multiply LR by gamma upon reaching a scheduled milestone')

# Noise injection
parser.add_argument('--corruption_type', type=str, choices=['unif', 'flip'])
parser.add_argument('--corruption_level', type=float, help='Corruption level')
parser.add_argument('--gold_fraction', default=0.02, type=float, help='Gold fraction')

# Hardware
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--n_gpus', default=1, type=int)

args = parser.parse_args()


def set_logging(rank):
    filename = '_'.join([
        args.dataset,
        str(args.corruption_level),
        args.corruption_type,
        args.runid,
        str(args.epochs),
        str(args.seed),
        str(args.data_seed),
    ])

    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('out', exist_ok=True)

    logfile = 'logs/' + filename + '.log'
    logger = get_logger(logfile, rank)

    logger.info('Arguments:')
    for k, v in vars(args).items():
        logger.info(f'{k}: {v}')
    logger.info('')

    return logger


def build_models(rank, dataset, num_classes):
    args.embedding_dim = 512 if dataset == 'cifar10' else 2048
    model_fn = generalized_resnet34 if dataset == 'cifar10' else generalized_resnet50

    # ✅ ALWAYS TRAIN FROM SCRATCH: ssl=False
    main_net = model_fn(num_classes, args, ssl=False)
    meta_backbone = model_fn(num_classes, args, ssl=False)

    meta_net = ResNetFeatures(meta_backbone)
    enhancer = TeacherEnhancer(num_classes, args.embedding_dim, args.label_embedding_dim, args.mlp_hidden_dim)

    main_net = main_net.to(rank)
    main_net = DDP(main_net, device_ids=[rank])

    meta_net = meta_net.to(rank)
    meta_net = DDP(meta_net, device_ids=[rank])

    enhancer = enhancer.to(rank)
    enhancer = DDP(enhancer, device_ids=[rank])

    return main_net, meta_net, enhancer


def run(rank):
    ddp_setup(rank, world_size=args.n_gpus, min_rank=args.gpuid)
    logger = set_logging(rank)

    filename = '_'.join([
        args.dataset,
        args.corruption_type,
        args.runid,
        str(args.epochs),
        str(args.seed),
        str(args.data_seed),
    ])
    exp_id = filename

    try:
        assert 0 <= args.gold_fraction <= 1, 'Wrong gold fraction!'
        assert 0 <= args.corruption_level <= 1, 'Wrong noise level!'

        gold_loader, silver_loader, valid_loader, test_loader, num_classes = prepare_data(
            args.gold_fraction, args.corruption_level, args
        )

        main_net, meta_net, enhancer = build_models(rank, args.dataset, num_classes)

        trainer = Trainer(
            rank, args, main_net, meta_net, enhancer,
            gold_loader, silver_loader, valid_loader, test_loader,
            num_classes, logger, exp_id
        )
        trainer.train()
        test_acc = trainer.final_eval()

        if rank == 0:
            results = {'method': test_acc}
            logger.info(' '.join([
                'Gold fraction:', str(args.gold_fraction),
                '| Corruption level:', str(args.corruption_level),
                '| Method acc:', str(results['method'])
            ]))
            logger.info('')
            with open('out/' + filename, 'wb') as f:
                pickle.dump(results, f)

        logger.info("Dumped results_ours in file: " + filename)

    finally:
        # Always clean up DDP
        if dist.is_available() and dist.is_initialized():
            destroy_process_group()


if __name__ == "__main__":
    gpus = range(args.gpuid, args.gpuid + args.n_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpus)

    # ✅ Avoid mp.spawn for single GPU to prevent duplicated noisy logs
    if args.n_gpus == 1:
        run(0)
    else:
        mp.spawn(run, nprocs=args.n_gpus)
