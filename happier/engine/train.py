from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import happier.lib as lib

from happier.engine.checkpoint import checkpoint
from happier.engine.accuracy_calculator import evaluate
from happier.engine.base_training_loop import base_training_loop


def train(
    config,
    log_dir,
    net,
    net_momentum,
    criterion,
    optimizer,
    scheduler,
    scaler,
    acc,
    train_dts,
    val_dts,
    test_dts,
    sampler,
    writer,
    restore_epoch,
):
    # """""""""""""""""" Iter over epochs """"""""""""""""""""""""""
    lib.LOGGER.info(f"Training of model {config.experience.experiment_name}")

    metrics = None
    for e in range(1 + restore_epoch, config.experience.max_iter + 1):

        for crit, _ in criterion:
            if hasattr(crit, "use_net") and crit.not_init:
                crit_loader = DataLoader(
                    train_dts,
                    num_workers=config.experience.num_workers,
                    pin_memory=config.experience.pin_memory,
                    batch_size=config.experience.accuracy_calculator.inference_batch_size,
                )
                crit.use_net(net, crit_loader)

        lib.LOGGER.info(f"Training : @epoch #{e} for model {config.experience.experiment_name}")
        start_time = time()

        # """""""""""""""""" Training Loop """"""""""""""""""""""""""
        sampler.reshuffle()
        loader = DataLoader(
            train_dts,
            batch_sampler=sampler,
            num_workers=config.experience.num_workers,
            pin_memory=config.experience.pin_memory,
        )
        logs = base_training_loop(
            config=config,
            net=net,
            net_momentum=net_momentum,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=e,
        )

        if (
            (config.experience.warmup_step is not None)
            and (config.experience.warmup_step >= e)
        ):
            pass
        else:
            for sch in scheduler["on_epoch"]:
                sch.step()

            for crit, _ in criterion:
                if hasattr(crit, 'on_epoch'):
                    crit.on_epoch()

        end_train_time = time()

        dataset_dict = {}
        if (config.experience.train_eval_freq > -1) and ((e % config.experience.train_eval_freq == 0) or (e == config.experience.max_iter)):
            dataset_dict["train"] = train_dts

        if (config.experience.val_eval_freq > -1) and ((e % config.experience.val_eval_freq == 0) or (e == config.experience.max_iter)):
            dataset_dict["val"] = val_dts

        if (config.experience.test_eval_freq > -1) and ((e % config.experience.test_eval_freq == 0) or (e == config.experience.max_iter)):
            if isinstance(test_dts, list):
                for i, _dts in enumerate(test_dts):
                    dataset_dict[f"test_level{i}"] = _dts
            else:
                dataset_dict["test"] = test_dts


        def local_evaluate(net, tb_name='Evaluation'):
            metrics = evaluate(
                net=net,
                dataset_dict=dataset_dict,
                acc=acc,
                epoch=e,
            )
            if metrics is not None:
                for split, mtrc in metrics.items():
                    for k, v in mtrc.items():
                        if k == 'epoch':
                            continue
                        lib.LOGGER.info(f"{split} --> {k} : {np.around(v * 100, decimals=2)}")
                        writer.add_scalar(f"NDCG/{split.title()}/"+tb_name+f"/{k}", v, e)
                    print()

            return metrics

        metrics = None
        if dataset_dict:
            metrics = local_evaluate(net)
            torch.cuda.empty_cache()

            if 'use_ema' in config.model.kwargs and config.model.kwargs.use_ema and \
                    'test_with_ema_net' in config.model.kwargs and config.model.kwargs.test_with_ema_net:
                _ = local_evaluate(net_momentum, tb_name='Evaluation_EMA')

            torch.cuda.empty_cache()


        # """""""""""""""""" Logging Step """"""""""""""""""""""""""
        for grp, opt in optimizer.items():
            writer.add_scalar(f"LR/{grp}", list(lib.get_lr(opt).values())[0], e)

        for k, v in logs.items():
            lib.LOGGER.info(f"{k} : {v:.4f}")
            writer.add_scalar(f"NDCG/Train/{k}", v, e)


        end_time = time()

        elapsed_time = lib.format_time(end_time - start_time)
        elapsed_time_train = lib.format_time(end_train_time - start_time)
        elapsed_time_eval = lib.format_time(end_time - end_train_time)

        lib.LOGGER.info(f"Epoch took : {elapsed_time}")
        if metrics is not None:
            lib.LOGGER.info(f"Training loop took : {elapsed_time_train}")
            lib.LOGGER.info(f"Evaluation step took : {elapsed_time_eval}")

        print()
        print()

        # """""""""""""""""" Checkpointing """"""""""""""""""""""""""
        checkpoint(
            log_dir=log_dir,
            save_checkpoint=(e % config.experience.save_model == 0) or (e == config.experience.max_iter),
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            epoch=e,
            config=config,
            metrics=metrics,
            net_momentum=net_momentum,
        )

    return metrics
