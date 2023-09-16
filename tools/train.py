# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.adadelta import Adadelta
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.criterion import CrossEntropy, OhemCrossEntropy
from core.function import train, validate
from external_utils.modelsummary import get_model_summary
from external_utils.utils import create_logger, FullModel, get_rank


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    # distributed = len(gpus) > 1
    local_rank = gpus[0]
    device = torch.device('cuda:{}'.format(local_rank))

    # build model
    model = models.seg_hrnet.get_seg_model(config)

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TRAIN_SET,
        num_samples=None,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=config.TRAIN.MULTI_SCALE,
        flip=config.TRAIN.FLIP,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TRAIN.BASE_SIZE,
        crop_size=crop_size,
        downsample_rate=config.TRAIN.DOWNSAMPLERATE,
        scale_factor=config.TRAIN.SCALE_FACTOR)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True)

    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_samples=config.TEST.NUM_SAMPLES,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
        center_crop_test=config.TEST.CENTER_CROP_TEST,
        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=train_dataset.class_weights)
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=train_dataset.class_weights)

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD([{'params':
                                          filter(lambda p: p.requires_grad,
                                                 model.parameters()),
                                      'lr': config.TRAIN.LR}],
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV,
                                    )
    elif config.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WD)
    elif config.TRAIN.OPTIMIZER == 'adadelta_frozen':
        # use adadelta and froze the low layers of network
        optimizer = Adadelta([{'params': model.conv1.parameters(), 'lr': config.TRAIN.LOW_LEVEL_LR * config.TRAIN.LR},
                              {'params': model.bn1.parameters(), 'lr': config.TRAIN.LOW_LEVEL_LR * config.TRAIN.LR},
                              {'params': model.conv2.parameters(), 'lr': config.TRAIN.LOW_LEVEL_LR * config.TRAIN.LR},
                              {'params': model.bn2.parameters(), 'lr': config.TRAIN.LOW_LEVEL_LR * config.TRAIN.LR},
                              {'params': model.layer1.parameters(), 'lr': config.TRAIN.LOW_LEVEL_LR * config.TRAIN.LR},
                              {'params': model.transition1.parameters(), 'lr': config.TRAIN.LOW_LEVEL_LR * config.TRAIN.LR},
                              {'params': model.stage2.parameters(), 'lr': config.TRAIN.LOW_LEVEL_LR * config.TRAIN.LR},
                              {'params': model.transition2.parameters(), 'lr': config.TRAIN.LOW_LEVEL_LR * config.TRAIN.LR},
                              {'params': model.stage3.parameters(), 'lr': config.TRAIN.LOW_LEVEL_LR * config.TRAIN.LR},
                              {'params': model.transition3.parameters(), 'lr': config.TRAIN.LOW_LEVEL_LR * config.TRAIN.LR},
                              {'params': model.stage4.parameters()},
                              {'params': model.last_layer.parameters()},
                              ],
                             lr=config.TRAIN.LR, weight_decay=config.TRAIN.WD)
    else:
        raise ValueError('Only Support SGD and Adam optimizer')

    model = FullModel(model, criterion)
    model = model.to(device)
    model = nn.parallel.DataParallel(model, device_ids=list(config.GPUS))

    epoch_iters = np.int(train_dataset.__len__() /
                         config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    best_mIoU = 0
    last_epoch = 0
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file,
                                    map_location=lambda storage, loc: storage)
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters

    for epoch in range(last_epoch, end_epoch):
        train(config, epoch, config.TRAIN.END_EPOCH,
              epoch_iters, config.TRAIN.LR, num_iters,
              trainloader, testloader, optimizer, model, writer_dict, device, final_output_dir)



        if epoch == end_epoch - 1:
            torch.save(model.module.state_dict(),
                       os.path.join(final_output_dir, 'final_state.pth'))

            writer_dict['writer'].close()
            end = timeit.default_timer()
            logger.info('Hours: %d' % np.int((end - start) / 3600))
            logger.info('Done')


if __name__ == '__main__':
    main()
