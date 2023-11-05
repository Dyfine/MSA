import os
from os.path import join
import random

import numpy as np
from omegaconf import OmegaConf
import hydra
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import happier.lib as lib
import happier.engine as eng
from happier.getter import Getter
import datetime
from happier.datasets.base_dataset import GaussianBlur, Solarization
from torchvision import transforms
from PIL import Image

def if_func(cond, x, y):
    if not isinstance(cond, bool):
        cond = eval(cond)
        assert isinstance(cond, bool)
    if cond:
        return x
    return y


OmegaConf.register_new_resolver("mult", lambda *numbers: np.prod([float(x) for x in numbers]))
OmegaConf.register_new_resolver("sum", lambda *numbers: sum(map(float, numbers)))
OmegaConf.register_new_resolver("sub", lambda x, y: float(x) - float(y))
OmegaConf.register_new_resolver("div", lambda x, y: float(x) / float(y))
OmegaConf.register_new_resolver("if", if_func)


@hydra.main(config_path='config', config_name='default')
def run(config):
    # torch.autograd.set_detect_anomaly(True)
    """
    creates all objects required to launch a training
    """
    # """""""""""""""""" Handle Config """"""""""""""""""""""""""
    config.experience.log_dir = lib.expand_path(config.experience.log_dir)
    now_time = datetime.datetime.now().strftime('%m%d_%H%M')
    log_dir = join(config.experience.log_dir, config.experience.experiment_name+'_'+now_time)

    if 'debug' in config.experience.experiment_name.lower():
        config.experience.DEBUG = config.experience.DEBUG or 1

    if config.experience.resume is not None:
        if os.path.isfile(lib.expand_path(config.experience.resume)):
            resume = lib.expand_path(config.experience.resume)
        else:
            resume = os.path.join(log_dir, 'weights', config.experience.resume)
            if not os.path.isfile(resume):
                lib.LOGGER.warning("Checkpoint does not exists")
                return

        state = torch.load(resume, map_location='cpu')
        at_epoch = state["epoch"]
        if at_epoch >= config.experience.max_iter:
            lib.LOGGER.warning(f"Exiting trial, experiment {config.experience.experiment_name} already finished")
            return

        lib.LOGGER.info(f"Resuming from state : {resume}")
        restore_epoch = state['epoch']

    else:
        resume = None
        state = None
        restore_epoch = 0
        if os.path.isdir(os.path.join(log_dir, 'weights')) and not config.experience.DEBUG:
            lib.LOGGER.warning(f"Exiting trial, experiment {config.experience.experiment_name} already exists")
            lib.LOGGER.warning(f"Its access: {log_dir}")
            return

    os.makedirs(join(log_dir, 'logs'), exist_ok=True)
    os.makedirs(join(log_dir, 'weights'), exist_ok=True)
    writer = SummaryWriter(join(log_dir, "logs"), purge_step=restore_epoch)

    # """""""""""""""""" Handle Reproducibility"""""""""""""""""""""""""
    lib.LOGGER.info(f"Training with seed {config.experience.seed}")
    random.seed(config.experience.seed)
    np.random.seed(config.experience.seed)
    torch.manual_seed(config.experience.seed)
    torch.cuda.manual_seed_all(config.experience.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # """""""""""""""""" Create Data """"""""""""""""""""""""""
    os.environ['USE_CUDA_FOR_RELEVANCE'] = 'yes'
    getter = Getter()


    train_transform = getter.get_transform(config.transform.train)
    test_transform = getter.get_transform(config.transform.test)
    train_dts = getter.get_dataset(train_transform, 'train', config.dataset)
    test_dts = getter.get_dataset(test_transform, 'test', config.dataset)
    val_dts = None

    sampler = getter.get_sampler(train_dts, config.dataset.sampler)

    # """""""""""""""""" Create Network """"""""""""""""""""""""""
    net = getter.get_model(config.model)

    if 'dino_pretrain_path' in config.model.kwargs.keys():
        dino_pretrain_path = config.model.kwargs['dino_pretrain_path']
    else:
        dino_pretrain_path = None
    if dino_pretrain_path:
        checkpoint = torch.load(os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'pretrained_model/' + dino_pretrain_path + '/ckpt.pth'),
            map_location='cpu')
        state_dict = checkpoint['student']
        for k in list(state_dict.keys()):
            if k.startswith('module.backbone.'):
                state_dict[k.replace('module.backbone.', '')] = state_dict[k]
            del state_dict[k]

        for k in list(state_dict.keys()):
            if k.startswith('conv1.'):
                state_dict[k.replace('conv1.', '0.')] = state_dict[k]
            elif k.startswith('bn1.'):
                state_dict[k.replace('bn1.', '1.')] = state_dict[k]
            elif k.startswith('layer1.'):
                state_dict[k.replace('layer1.', '4.')] = state_dict[k]
            elif k.startswith('layer2.'):
                state_dict[k.replace('layer2.', '5.')] = state_dict[k]
            elif k.startswith('layer3.'):
                state_dict[k.replace('layer3.', '6.')] = state_dict[k]
            elif k.startswith('layer4.'):
                state_dict[k.replace('layer4.', '7.')] = state_dict[k]
            del state_dict[k]

        net.backbone.load_state_dict(state_dict, strict=True)
        lib.LOGGER.info(f"load dino pretrained weights: {dino_pretrain_path}")


    scaler = None
    if config.model.kwargs.with_autocast:
        scaler = torch.cuda.amp.GradScaler()
        if state is not None:
            scaler.load_state_dict(state['scaler_state'])

    if state is not None:
        net.load_state_dict(state['net_state'])
        net.cuda()

    net_momentum = None
    if 'use_ema' in config.model.kwargs and config.model.kwargs.use_ema:
        net_momentum = getter.get_model(config.model)
        lib.momentum_update(net, net_momentum, m=0)

        if dino_pretrain_path:
            checkpoint = torch.load(os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'pretrained_model/' + dino_pretrain_path + '/ckpt.pth'),
                map_location='cpu')
            state_dict = checkpoint['teacher']
            for k in list(state_dict.keys()):
                if k.startswith('module.backbone.'):
                    state_dict[k.replace('module.backbone.', '')] = state_dict[k]
                del state_dict[k]

            for k in list(state_dict.keys()):
                if k.startswith('conv1.'):
                    state_dict[k.replace('conv1.', '0.')] = state_dict[k]
                elif k.startswith('bn1.'):
                    state_dict[k.replace('bn1.', '1.')] = state_dict[k]
                elif k.startswith('layer1.'):
                    state_dict[k.replace('layer1.', '4.')] = state_dict[k]
                elif k.startswith('layer2.'):
                    state_dict[k.replace('layer2.', '5.')] = state_dict[k]
                elif k.startswith('layer3.'):
                    state_dict[k.replace('layer3.', '6.')] = state_dict[k]
                elif k.startswith('layer4.'):
                    state_dict[k.replace('layer4.', '7.')] = state_dict[k]
                del state_dict[k]

            net_momentum.backbone.load_state_dict(state_dict, strict=True)
            lib.LOGGER.info(f"load dino pretrained weights for net_momentum")

        for param_k in net_momentum.parameters():
            param_k.requires_grad = False  # not update by gradient

    # """""""""""""""""" Create Optimizer & Scheduler """"""""""""""""""""""""""
    optimizer, scheduler = getter.get_optimizer(net, config.optimizer)

    if state is not None:
        for key, opt in optimizer.items():
            opt.load_state_dict(state['optimizer_state'][key])

        for key, sch in scheduler.items():
            sch.load_state_dict(state[f'scheduler_{key}_state'])

    # """""""""""""""""" Create Criterion """"""""""""""""""""""""""
    criterion = getter.get_loss(config.loss)

    for crit, _ in criterion:
        if hasattr(crit, 'register_labels'):
            crit.register_labels(torch.from_numpy(train_dts.labels))

    if state is not None and "criterion_state" in state:
        for (crit, _), crit_state in zip(criterion, state["criterion_state"]):
            crit.cuda()
            crit.load_state_dict(crit_state)

    acc = getter.get_acc_calculator(config.experience)

    # """""""""""""""""" Handle Cuda """"""""""""""""""""""""""
    if torch.cuda.device_count() > 1:
        lib.LOGGER.info("Model is parallelized")
        net = nn.DataParallel(net)
        if net_momentum is not None:
            net_momentum = nn.DataParallel(net_momentum)

    if config.experience.parallelize_loss:
        for i, (crit, w) in enumerate(criterion):
            level = crit.hierarchy_level
            crit = nn.DataParallel(crit)
            crit.hierarchy_level = level

    net.cuda()
    if net_momentum is not None:
        net_momentum.cuda()
    _ = [crit.cuda() for crit, _ in criterion]

    # """""""""""""""""" Handle RANDOM_STATE """"""""""""""""""""""""""
    if state is not None:
        # set random NumPy and Torch random states
        lib.set_random_state(state)

    return eng.train(
        config=config,
        log_dir=log_dir,
        net=net,
        net_momentum=net_momentum,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        acc=acc,
        train_dts=train_dts,
        val_dts=val_dts,
        test_dts=test_dts,
        sampler=sampler,
        writer=writer,
        restore_epoch=restore_epoch,
    )


if __name__ == '__main__':
    run()
