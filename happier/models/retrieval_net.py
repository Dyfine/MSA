import torch
import torch.nn as nn
import torch.nn.functional as F

import happier.lib as lib

from happier.models.get_pooling import get_pooling
from happier.models.get_backbone import get_backbone
import math

def flatten(tens):
    if tens.ndim == 2:
        return tens.squeeze(1)
    if tens.ndim == 3:
        return tens.squeeze(2).squeeze(1)
    if tens.ndim == 4:
        return tens.squeeze(3).squeeze(2).squeeze(1)



class RetrievalNet(nn.Module):

    def __init__(
        self,
        backbone_name,
        embed_dim=512,
        normalize=True,
        norm_features=False,
        without_fc=False,
        with_autocast=True,
        pooling='default',
        projection_normalization_layer='none',
        pretrained=True,
        **kwargs,
    ):
        super().__init__()

        norm_features = lib.str_to_bool(norm_features)
        without_fc = lib.str_to_bool(without_fc)
        with_autocast = lib.str_to_bool(with_autocast)

        self.embed_dim = embed_dim
        self.normalize = normalize
        self.norm_features = norm_features
        self.without_fc = without_fc
        self.with_autocast = with_autocast
        if with_autocast:
            lib.LOGGER.info("Using mixed precision")

        self.backbone, default_pooling, out_features = get_backbone(backbone_name, pretrained=pretrained, **kwargs)
        self.pooling = get_pooling(default_pooling, pooling)
        lib.LOGGER.info(f"Pooling is {self.pooling}")

        if self.norm_features:
            lib.LOGGER.info("Using a LayerNorm layer")
            self.standardize = nn.LayerNorm(out_features, elementwise_affine=False)
        else:
            self.standardize = nn.Identity()

        if not self.without_fc:
            self.fc = nn.Linear(out_features, embed_dim)
            lib.LOGGER.info(f"Projection head : \n{self.fc}")
        else:
            self.fc = nn.Identity()
            lib.LOGGER.info("Not using a linear projection layer")

        if 'use_sep' in kwargs.keys():
            self.use_sep = kwargs['use_sep']
        else:
            self.use_sep = False

        if self.use_sep:
            self.standardize = nn.LayerNorm(embed_dim, elementwise_affine=False)

        self.fc_aux = nn.Linear(out_features, embed_dim)

    def forward(self, X):
        with torch.cuda.amp.autocast(enabled=self.with_autocast or (not self.training)):
            X = self.backbone(X)
            X = self.pooling(X)
            X = flatten(X)

            X_disti = self.fc_aux(X)

            if not self.use_sep:
                X_sep = self.standardize(X)
                X_sep = self.fc(X_sep)
            else:
                X_sep = self.fc(X)
                X_sep = self.standardize(X_sep)

            if self.normalize or (not self.training):
                X = X_sep
                if self.use_sep:
                    if not self.training:
                        B, C = X.size()
                        emb0 = F.normalize(X[:, :C // 3], dim=1)
                        emb1 = F.normalize(X[:, C // 3: C // 3 * 2], dim=1)
                        emb2 = F.normalize(X[:, C // 3 * 2:], dim=1)
                        X = torch.cat((emb0, emb1, emb2), dim=1) / (3**0.5)
                else:
                    dtype = X.dtype
                    X = F.normalize(X, p=2, dim=-1).to(dtype)

                return X

            return X_sep, X_disti
