import os

import torch
from tqdm import tqdm

import happier.lib as lib
import math


def _calculate_loss_and_backward(
    config,
    net,
    net_momentum,
    batch,
    relevance_fn,
    criterion,
    optimizer,
    scaler,
    epoch,
):
    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
        if config.loss.name in ['ClusterLoss_MultiEmb', 'ClusterLoss_SingleEmb']:
            di, di_aux = net(batch["image"].cuda())
        else:
            di = net(batch["image"].cuda())
            di_aux = None
        labels = batch["label"].cuda()

        di_ema = None
        if net_momentum is not None:
            with torch.no_grad():
                if config.model.kwargs.ema_same_aug:
                    _, di_ema = net_momentum(batch["image"].cuda())
                else:
                    _, di_ema = net_momentum(batch["image1"].cuda())

        logs = {}
        losses = []
        for crit, weight in criterion:
            loss = crit(
                di,
                labels,
                relevance_fn=relevance_fn,
                indexes=batch["index"].cuda(),
                epoch=epoch,
                max_epoch=config.experience.max_iter,
                emb_ema=di_ema,
                di_aux=di_aux,
            )

            if isinstance(loss, dict):
                for k,v in loss.items():
                    if 'total_loss' not in k:
                        logs[k] = v.item()
                loss = loss['total_loss']
            else:
                logs[f"{crit.__class__.__name__}_l{crit.hierarchy_level}"] = loss.item()

            loss = loss.mean()
            losses.append(weight * loss)

            # logs[f"{crit.__class__.__name__}_l{crit.hierarchy_level}"] = loss.item()

    total_loss = sum(losses)
    if scaler is None:
        total_loss.backward()
    else:
        scaler.scale(total_loss).backward()

    logs["total_loss"] = total_loss.item()
    _ = [loss.detach_() for loss in losses]
    total_loss.detach_()
    return logs


def base_training_loop(
    config,
    net,
    net_momentum,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    epoch,
):
    meter = lib.DictAverage()
    net.train()
    net.zero_grad()

    iterator = tqdm(loader, disable=os.getenv('TQDM_DISABLE'))
    for i, batch in enumerate(iterator):
        logs = _calculate_loss_and_backward(
            config,
            net,
            net_momentum,
            batch,
            loader.dataset.compute_relevance_on_the_fly,
            criterion,
            optimizer,
            scaler,
            epoch,
        )

        if config.experience.record_gradient:
            if scaler is not None:
                for opt in optimizer.values():
                    scaler.unscale_(opt)

            logs["gradient_norm"] = lib.get_gradient_norm(net)

        if config.experience.gradient_clipping_norm is not None:
            if (scaler is not None) and (not config.experience.record_gradient):
                for opt in optimizer.values():
                    scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.experience.gradient_clipping_norm)

        for key, opt in optimizer.items():
            if (
                (config.experience.warmup_step is not None)
                and (config.experience.warmup_step >= epoch)
                and (key in config.experience.warmup_keys)
            ):
                if i == 0:
                    lib.LOGGER.warning("Warmimg UP")
                continue
            if scaler is None:
                opt.step()
            else:
                scaler.step(opt)

        for crit, _ in criterion:
            if hasattr(crit, 'update'):
                crit.update(scaler)

        net.zero_grad()
        _ = [crit.zero_grad() for crit, w in criterion]

        for sch in scheduler["on_step"]:
            sch.step()

        if scaler is not None:
            scaler.update()

        meter.update(logs)
        if not os.getenv('TQDM_DISABLE'):
            iterator.set_postfix(meter.avg)
        else:
            if (i + 1) % config.experience.print_freq == 0:
                lib.LOGGER.info(f'Iteration : {i}/{len(loader)}')
                for k, v in logs.items():
                    lib.LOGGER.info(f'Loss: {k}: {v} ')

        if config.experience.DEBUG:
            if isinstance(config.experience.DEBUG, int):
                if (i+1) > config.experience.DEBUG:
                    break
            else:
                break

        if net_momentum is not None:
            if 'ema_start_epoch' in config.model.kwargs and config.model.kwargs.ema_start_epoch is not None:
                ema_start_epoch = config.model.kwargs.ema_start_epoch
            else:
                ema_start_epoch = 1

            if epoch < ema_start_epoch:
                m = 1
            elif 'ema_m_increase' in config.model.kwargs and config.model.kwargs.ema_m_increase:
                # linear
                # m = ((epoch-1)*len(loader) + i) / (config.experience.max_iter * len(loader)) * (1 - config.model.kwargs.ema_m) + config.model.kwargs.ema_m
                # cosine
                ratio = ((epoch-ema_start_epoch)*len(loader) + i) / ((config.experience.max_iter-ema_start_epoch+1) * len(loader))
                m = (-math.cos(ratio * math.pi) + 1) / 2 * (1 - config.model.kwargs.ema_m) + config.model.kwargs.ema_m
            else:
                m = config.model.kwargs.ema_m
            lib.momentum_update(net, net_momentum, m=m)


    for crit, _ in criterion:
        if hasattr(crit, 'optimize_proxies'):
            crit.optimize_proxies(loader.dataset.compute_relevance_on_the_fly)

    return meter.avg
