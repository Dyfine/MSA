import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import happier.lib as lib
import numpy as np

# adapted from :
# https://github.com/azgo14/classification_metric_learning/blob/master/metric_learning/modules/losses.py
class ClusterLoss(nn.Module):
    """
    L2 normalize weights and apply temperature scaling on logits.
    """
    def __init__(
        self,
        embedding_size,
        num_classes,
        temperature=0.05,
        hierarchy_level=None,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.temperature = temperature
        self.hierarchy_level = hierarchy_level

        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings, instance_targets, relevance_fn=None, **kwargs,):
        if self.hierarchy_level is not None:
            instance_targets = instance_targets[:, self.hierarchy_level]

        norm_weight = F.normalize(self.weight, dim=1)

        prediction_logits = F.linear(embeddings, norm_weight)

        loss = self.loss_fn(prediction_logits / self.temperature, instance_targets)
        return loss

    def register_optimizers(self, opt, sch):
        self.opt = opt
        self.sch = sch
        lib.LOGGER.info(f"Optimizer registered for {self.__class__.__name__}")

    def update(self, scaler=None):
        if scaler is None:
            self.opt.step()
        else:
            scaler.step(self.opt)

        if self.sch["on_step"]:
            self.sch["on_step"].step()

    def on_epoch(self,):
        if self.sch["on_epoch"]:
            self.sch["on_epoch"].step()

    def on_val(self, val):
        if self.sch["on_val"]:
            self.sch["on_val"].step(val)

    def state_dict(self, *args, **kwargs):
        state = {"super": super().state_dict(*args, **kwargs)}
        state["opt"] = self.opt.state_dict()
        state["sch_on_step"] = self.sch["on_step"].state_dict() if self.sch["on_step"] else None
        state["sch_on_epoch"] = self.sch["on_epoch"].state_dict() if self.sch["on_epoch"] else None
        state["sch_on_val"] = self.sch["on_val"].state_dict() if self.sch["on_val"] else None
        return state

    def load_state_dict(self, state_dict, override=False, *args, **kwargs):
        super().load_state_dict(state_dict["super"], *args, **kwargs)
        if not override:
            self.opt.load_state_dict(state_dict["opt"])
            if self.sch["on_step"]:
                self.sch["on_step"].load_state_dict(state_dict["sch_on_step"])
            if self.sch["on_epoch"]:
                self.sch["on_epoch"].load_state_dict(state_dict["sch_on_epoch"])
            if self.sch["on_val"]:
                self.sch["on_val"].load_state_dict(state_dict["sch_on_val"])

    def __repr__(self,):
        repr = f"{self.__class__.__name__}(\n"
        repr = repr + f"    temperature={self.temperature},\n"
        repr = repr + f"    num_classes={self.num_classes},\n"
        repr = repr + f"    embedding_size={self.embedding_size},\n"
        repr = repr + f"    opt={self.opt.__class__.__name__},\n"
        repr = repr + f"    hierarchy_level={self.hierarchy_level},\n"
        repr = repr + ")"
        return repr



class ClusterLoss_MultiEmb(nn.Module):

    def __init__(
            self,
            embedding_size,
            num_classes_level0,
            num_classes_level1,
            num_classes_level2,
            data_dir=None,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes_level0 = num_classes_level0
        self.num_classes_level1 = num_classes_level1
        self.num_classes_level2 = num_classes_level2

        num_split = 3
        self.dim_f, self.dim_m, self.dim_c = embedding_size//num_split, embedding_size//num_split, embedding_size//num_split
        

        # 0: fine, 1: mid, 2: coarse
        data_name = data_dir.strip().split('_')[-1]
        if data_name == 'vehicle':
            self.num_proxy0 = 2
            self.num_proxy1 = 20
            self.num_proxy2 = 4
            self.m_c = 0.1
            self.s_c = 20
            self.alpha = 0.25
            self.beta = 0.75
        elif data_name == 'animal':
            self.num_proxy0 = 28
            self.num_proxy1 = 4
            self.num_proxy2 = 2
            self.m_c = 0.
            self.s_c = 20
            self.alpha = 0.25
            self.beta = 0.75
        elif data_name == 'product':
            self.num_proxy0 = 22
            self.num_proxy1 = 3
            self.num_proxy2 = 2
            self.m_c = 0.
            self.s_c = 20
            self.alpha = 0.25
            self.beta = 0.1


        self.weight_level0m = nn.Parameter(torch.Tensor(num_classes_level0 * self.num_proxy0, self.dim_f))
        self.weight_level1m = nn.Parameter(torch.Tensor(num_classes_level1 * self.num_proxy1, self.dim_m))
        self.weight_level2m = nn.Parameter(torch.Tensor(num_classes_level2 * self.num_proxy2, self.dim_c))
        stdv = 1. / math.sqrt(self.weight_level0m.size(1))
        self.weight_level0m.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_level1m.size(1))
        self.weight_level1m.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_level2m.size(1))
        self.weight_level2m.data.uniform_(-stdv, stdv)

        self.loss_fn = nn.CrossEntropyLoss()

        self.online_linear = nn.Linear(embedding_size, embedding_size)
        self.fc_f = nn.Linear(self.dim_f, self.dim_f)
        self.fc_m = nn.Linear(self.dim_m, self.dim_m)
        self.fc_c = nn.Linear(self.dim_c, self.dim_c)


    def forward(self, embeddings, instance_targets, relevance_fn=None, **kwargs, ):
        loss_cluster, loss_cluster0, loss_cluster1, loss_cluster2 = self.forward_single_level_loss(embeddings, instance_targets, relevance_fn, **kwargs)

        loss_cross_level = self.forward_cross_level_constraint(embeddings, instance_targets, relevance_fn, **kwargs)

        loss_instance_alignment = self.forward_instance_alignment(embeddings, instance_targets, relevance_fn, **kwargs)

        loss = loss_cluster + self.alpha * loss_cross_level + self.beta * loss_instance_alignment


        loss_dict = {'total_loss': loss}
        loss_dict['loss_cluster'] = loss_cluster
        loss_dict['loss_cluster0'] = loss_cluster0
        loss_dict['loss_cluster1'] = loss_cluster1
        loss_dict['loss_cluster2'] = loss_cluster2
        loss_dict['loss_cross_level'] = loss_cross_level * self.alpha
        loss_dict['loss_instance_alignment'] = loss_instance_alignment * self.beta

        return loss_dict


    def forward_single_level_loss(self, embeddings, instance_targets, relevance_fn=None, **kwargs,):
        epoch = kwargs['epoch']
        B, C = embeddings.size()
        device = embeddings.device

        instance_targets0 = instance_targets[:, 0]
        instance_targets1 = instance_targets[:, 1]
        instance_targets2 = instance_targets[:, 2]
        

        norm_weight0m = F.normalize(self.weight_level0m, dim=1)
        emb0 = embeddings[:, : self.dim_f]
        emb0 = F.normalize(emb0, dim=1)
        sim0 = torch.mm(emb0, norm_weight0m.t())
        sim0 = sim0.reshape(B, self.num_classes_level0, self.num_proxy0)
        sim0_prob = F.softmax(sim0 * 10., dim=2)
        sim0 = torch.sum(sim0 * sim0_prob, dim=2)
        # sim0 = torch.max(sim0, dim=2)[0]
        mask_p0 = torch.zeros_like(sim0).scatter_(dim=1, index=instance_targets0.view(-1, 1),
                                                  src=torch.ones_like(sim0)).detach()
        logits0 = (sim0 - self.m_c * mask_p0.float()) * self.s_c
        loss0 = self.loss_fn(logits0, instance_targets0)
        

        norm_weight1m = F.normalize(self.weight_level1m, dim=1)
        emb1 = embeddings[:, self.dim_f : self.dim_f+self.dim_m]
        emb1 = F.normalize(emb1, dim=1)
        sim1 = torch.mm(emb1, norm_weight1m.t())
        sim1 = sim1.reshape(B, self.num_classes_level1, self.num_proxy1)
        sim1_prob = F.softmax(sim1 * 10., dim=2)
        sim1 = torch.sum(sim1 * sim1_prob, dim=2)
        # sim1 = torch.max(sim1, dim=2)[0]
        mask_p1 = torch.zeros_like(sim1).scatter_(dim=1, index=instance_targets1.view(-1, 1),
                                                  src=torch.ones_like(sim1)).detach()
        logits1 = (sim1 - self.m_c * mask_p1.float()) * self.s_c
        loss1 = self.loss_fn(logits1, instance_targets1)


        norm_weight2m = F.normalize(self.weight_level2m, dim=1)
        emb2 = embeddings[:, self.dim_f+self.dim_m :]
        emb2 = F.normalize(emb2, dim=1)
        sim2 = torch.mm(emb2, norm_weight2m.t())
        sim2 = sim2.reshape(B, self.num_classes_level2, self.num_proxy2)
        sim2_prob = F.softmax(sim2 * 10., dim=2)
        sim2 = torch.sum(sim2 * sim2_prob, dim=2)
        # sim2 = torch.max(sim2, dim=2)[0]
        mask_p2 = torch.zeros_like(sim2).scatter_(dim=1, index=instance_targets2.view(-1, 1),
                                                  src=torch.ones_like(sim2)).detach()
        logits2 = (sim2 - self.m_c * mask_p2.float()) * self.s_c
        loss2 = self.loss_fn(logits2, instance_targets2)
        

        loss = (loss0 + loss1 + loss2) / 3

        return loss, loss0, loss1, loss2

    def forward_cross_level_constraint(self, embeddings, instance_targets, relevance_fn=None, **kwargs, ):
        instance_targets0 = instance_targets[:, 0]
        instance_targets1 = instance_targets[:, 1]
        instance_targets2 = instance_targets[:, 2]

        device = embeddings.device
        emb = embeddings
        B, C = emb.size()

        emb0 = F.normalize(emb[:, : self.dim_f], dim=1)
        emb1 = F.normalize(emb[:, self.dim_f : self.dim_f+self.dim_m], dim=1)
        emb2 = F.normalize(emb[:, self.dim_f+self.dim_m :], dim=1)

        mask = 1. - torch.eye(B).to(device).float()
        sim_f = torch.mm(emb0, emb0.t())
        sim_m = torch.mm(emb1, emb1.t())
        sim_c = torch.mm(emb2, emb2.t())

        mask_f_pos = instance_targets0.view(-1, 1) == instance_targets0.view(1, -1)
        mask_m_pos = instance_targets1.view(-1, 1) == instance_targets1.view(1, -1)
        mask_c_pos = instance_targets2.view(-1, 1) == instance_targets2.view(1, -1)

        
        min_m_c = torch.minimum(sim_m, sim_c)
        sim_f_changed = (mask_f_pos.float() * (min_m_c < sim_f).float().detach()) == 1

        sim_m_changed = mask_m_pos.float() * (sim_c < sim_m).float().detach() + (1 - mask_m_pos.float()) * (
                    sim_f > sim_m).float().detach()
        sim_m_changed = sim_m_changed == 1

        max_f_m = torch.maximum(sim_f, sim_m)
        sim_c_changed = ((1 - mask_c_pos.float()) * (max_f_m>sim_c).float().detach()) == 1

        
        con_f = mask_f_pos.float() * ((1. - sim_f)**2) + (1. - mask_f_pos.float()) * ((sim_f + 1)**2)
        con_m = mask_m_pos.float() * ((1. - sim_m)**2) + (1. - mask_m_pos.float()) * ((sim_m + 1)**2)
        con_c = mask_c_pos.float() * ((1. - sim_c)**2) + (1. - mask_c_pos.float()) * ((sim_c + 1)**2)
        loss_f = (con_f * (1 - sim_f_changed.float())).mean()
        loss_m = (con_m * (1 - sim_m_changed.float())).mean()
        loss_c = (con_c * (1 - sim_c_changed.float())).mean()

        
        loss_f1 = (mask_f_pos.float() * (sim_m<sim_f).detach().float() * con_m).sum() + \
                 (mask_f_pos.float() * (sim_c<sim_f).detach().float() * con_c).sum()
        loss_f1 = loss_f1 / (B*B)
        loss_f = loss_f + loss_f1
        

        mask_m_neg = ~mask_m_pos
        loss_m1 = (mask_m_pos.float() * (sim_c<sim_m).detach().float() * con_c).sum() + \
                 (mask_m_neg.float() * (sim_f>sim_m).detach().float() * con_f).sum()
        loss_m1 = loss_m1 / (B*B)
        loss_m = loss_m + loss_m1
        

        mask_c_neg = ~mask_c_pos
        loss_c1 = (mask_c_neg.float() * (sim_m>sim_c).detach().float() * con_m).sum() + \
                 (mask_c_neg.float() * (sim_f>sim_c).detach().float() * con_f).sum()
        loss_c1 = loss_c1 / (B*B)
        loss_c = loss_c + loss_c1
        

        return (loss_f + loss_m + loss_c) / 3

    def forward_instance_alignment(self, embeddings, instance_targets, relevance_fn=None, **kwargs, ):
        emb_ema = kwargs['emb_ema'].detach()
        emb = kwargs['di_aux']

        T, T_ema = 1., 1.

        emb_ema = F.normalize(emb_ema, p=2, dim=1)
        emb = F.normalize(emb, p=2, dim=1)
        p = F.softmax(emb_ema/T_ema, dim=1)
        log_q = F.log_softmax(emb/T, dim=1)
        loss = (-p * log_q).sum(dim=1).mean()

        return loss

    def register_optimizers(self, opt, sch):
        self.opt = opt
        self.sch = sch
        lib.LOGGER.info(f"Optimizer registered for {self.__class__.__name__}")

    def register_labels(self, labels):
        assert self.num_classes_level0 == len(labels[:, 0].unique())
        assert self.num_classes_level1 == len(labels[:, 1].unique())
        assert self.num_classes_level2 == len(labels[:, 2].unique())
        lib.LOGGER.info(f"Labels registered for {self.__class__.__name__}")


    def update(self, scaler=None):
        if scaler is None:
            self.opt.step()
        else:
            scaler.step(self.opt)

        if self.sch["on_step"]:
            self.sch["on_step"].step()

    def on_epoch(self,):
        if self.sch["on_epoch"]:
            self.sch["on_epoch"].step()

    def on_val(self, val):
        if self.sch["on_val"]:
            self.sch["on_val"].step(val)

    def state_dict(self, *args, **kwargs):
        state = {"super": super().state_dict(*args, **kwargs)}
        state["opt"] = self.opt.state_dict()
        state["sch_on_step"] = self.sch["on_step"].state_dict() if self.sch["on_step"] else None
        state["sch_on_epoch"] = self.sch["on_epoch"].state_dict() if self.sch["on_epoch"] else None
        state["sch_on_val"] = self.sch["on_val"].state_dict() if self.sch["on_val"] else None
        return state

    def load_state_dict(self, state_dict, override=False, *args, **kwargs):
        super().load_state_dict(state_dict["super"], *args, **kwargs)
        if not override:
            self.opt.load_state_dict(state_dict["opt"])
            if self.sch["on_step"]:
                self.sch["on_step"].load_state_dict(state_dict["sch_on_step"])
            if self.sch["on_epoch"]:
                self.sch["on_epoch"].load_state_dict(state_dict["sch_on_epoch"])
            if self.sch["on_val"]:
                self.sch["on_val"].load_state_dict(state_dict["sch_on_val"])

    def __repr__(self,):
        repr = f"{self.__class__.__name__}(\n"
        repr = repr + f"    embedding_size={self.embedding_size},\n"
        repr = repr + f"    opt={self.opt.__class__.__name__},\n"
        repr = repr + ")"
        return repr



class ClusterLoss_SingleEmb(nn.Module):
    """
    L2 normalize weights and apply temperature scaling on logits.
    """
    def __init__(
        self,
        embedding_size,
        num_classes_level0,
        num_classes_level1,
        num_classes_level2,
        data_dir,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes_level0 = num_classes_level0
        self.num_classes_level1 = num_classes_level1
        self.num_classes_level2 = num_classes_level2

        data_name = data_dir.strip().split('_')[-1]
        if data_name == 'product':
            self.num_proxy0 = 22
            self.m_c = 0.
            self.s_c = 20
            self.beta = 0.1


        self.weight_level0m = nn.Parameter(torch.Tensor(num_classes_level0 * self.num_proxy0, embedding_size))
        stdv = 1. / math.sqrt(self.weight_level0m.size(1))
        self.weight_level0m.data.uniform_(-stdv, stdv)

        self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, embeddings, instance_targets, epoch, relevance_fn=None, **kwargs, ):
        loss_cluster0 = self.forward_single_level_loss_level0(embeddings, instance_targets, relevance_fn, **kwargs)

        loss_instance_alignmnet = self.forward_instance_alignmnet(embeddings, instance_targets, relevance_fn, **kwargs)


        loss = loss_cluster0 + loss_instance_alignmnet * self.beta

        loss_dict = {'total_loss': loss}
        loss_dict['loss_cluster0'] = loss_cluster0
        loss_dict['loss_instance_alignmnet'] = loss_instance_alignmnet * self.beta

        return loss_dict


    def forward_single_level_loss_level0(self, embeddings, instance_targets, relevance_fn=None, **kwargs,):
        instance_targets0 = instance_targets[:, 0]
        B, C = embeddings.size()
        device = embeddings.device

        norm_weight0m = F.normalize(self.weight_level0m, dim=1)
        emb0 = F.normalize(embeddings, dim=1)
        sim0 = torch.mm(emb0, norm_weight0m.t())
        sim0 = sim0.reshape(B, self.num_classes_level0, self.num_proxy0)
        sim0_prob = F.softmax(sim0 * 10., dim=2)
        sim0 = torch.sum(sim0 * sim0_prob, dim=2)
        # sim0 = torch.max(sim0, dim=2)[0]
        mask_p0 = torch.zeros_like(sim0).scatter_(dim=1, index=instance_targets0.view(-1, 1),
                                                  src=torch.ones_like(sim0)).detach()
        logits0 = (sim0 - self.m_c * mask_p0.float()) * self.s_c
        loss = self.loss_fn(logits0, instance_targets0)

        return loss

    def forward_instance_alignmnet(self, embeddings, instance_targets, relevance_fn=None, **kwargs, ):
        emb_ema = kwargs['emb_ema'].detach()
        emb = kwargs['di_aux']

        T, T_ema = 1., 1.

        emb_ema = F.normalize(emb_ema, p=2, dim=1)
        emb = F.normalize(emb, p=2, dim=1)
        p = F.softmax(emb_ema/T_ema, dim=1)
        log_q = F.log_softmax(emb/T, dim=1)
        loss = (-p * log_q).sum(dim=1).mean()

        return loss

    def register_optimizers(self, opt, sch):
        self.opt = opt
        self.sch = sch
        lib.LOGGER.info(f"Optimizer registered for {self.__class__.__name__}")

    def register_labels(self, labels):
        assert self.num_classes_level0 == len(labels[:, 0].unique())
        assert self.num_classes_level1 == len(labels[:, 1].unique())
        assert self.num_classes_level2 == len(labels[:, 2].unique())
        lib.LOGGER.info(f"Labels registered for {self.__class__.__name__}")

    def update(self, scaler=None):
        if scaler is None:
            self.opt.step()
        else:
            scaler.step(self.opt)

        if self.sch["on_step"]:
            self.sch["on_step"].step()

    def on_epoch(self,):
        if self.sch["on_epoch"]:
            self.sch["on_epoch"].step()

    def on_val(self, val):
        if self.sch["on_val"]:
            self.sch["on_val"].step(val)

    def state_dict(self, *args, **kwargs):
        state = {"super": super().state_dict(*args, **kwargs)}
        state["opt"] = self.opt.state_dict()
        state["sch_on_step"] = self.sch["on_step"].state_dict() if self.sch["on_step"] else None
        state["sch_on_epoch"] = self.sch["on_epoch"].state_dict() if self.sch["on_epoch"] else None
        state["sch_on_val"] = self.sch["on_val"].state_dict() if self.sch["on_val"] else None
        return state

    def load_state_dict(self, state_dict, override=False, *args, **kwargs):
        super().load_state_dict(state_dict["super"], *args, **kwargs)
        if not override:
            self.opt.load_state_dict(state_dict["opt"])
            if self.sch["on_step"]:
                self.sch["on_step"].load_state_dict(state_dict["sch_on_step"])
            if self.sch["on_epoch"]:
                self.sch["on_epoch"].load_state_dict(state_dict["sch_on_epoch"])
            if self.sch["on_val"]:
                self.sch["on_val"].load_state_dict(state_dict["sch_on_val"])

    def __repr__(self,):
        repr = f"{self.__class__.__name__}(\n"
        repr = repr + f"    embedding_size={self.embedding_size},\n"
        repr = repr + f"    opt={self.opt.__class__.__name__},\n"
        repr = repr + ")"
        return repr
