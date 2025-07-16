""" timm model adapter

Wraps timm (https://github.com/rwightman/pytorch-image-models) models for use as a vision tower in CLIP model.
"""
import logging
from collections import OrderedDict
import pdb
from timm.models.registry import register_model
from timm.models.vision_transformer import VisionTransformer
from .dofa_patch_embed import DOFA_PatchEmbed

import torch
import torch.nn as nn

try:
    import timm
    try:
        # new timm imports >= 0.8.1
        from timm.layers import RotAttentionPool2d
        from timm.layers import AttentionPool2d as AbsAttentionPool2d
        from timm.layers import Mlp, to_2tuple
    except ImportError as e:
        # fallback, try old timm imports < 0.8.1
        from timm.models.layers.attention_pool2d import RotAttentionPool2d
        from timm.models.layers.attention_pool2d import AttentionPool2d as AbsAttentionPool2d
        from timm.models.layers import Mlp, to_2tuple
except ImportError:
    timm = None

from .utils import freeze_batch_norm_2d


class GeoLB_VisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, *args, **kwargs):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim, *args, **kwargs)
        # Replace the original patch embedding
        self.patch_embed = DOFA_PatchEmbed(wv_planes=128, inter_dim=128, kernel_size=patch_size, embed_dim=embed_dim)
        self.head = nn.Identity()
        self.fc_norm = nn.Identity()

    def forward_features(self, x, wvs=None):
        x, waves_tensor = self.patch_embed(x, wvs)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        #'''
        outx = []
        length = len(self.blocks)
        for i,block in enumerate(self.blocks):
            x = block(x)
            if i in [11,15,23,length-1]:
                outx.append(x)
        #'''
        #x = self.blocks(x)
        x = self.norm(x)
        return x, outx

    def forward(self, x, wvs):
        #if wvs is None:
        #    wvs = torch.tensor([0.665, 0.560, 0.490], device=x.device).float()
        x, outx = self.forward_features(x, wvs)
        sfeats = outx
        x = self.attn_pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        x = self.head(x)
        return x, sfeats

@register_model
def dofa_vit_base_patch16_siglip_224(pretrained=False, **kwargs):
    #img_size=224 by default
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
    )
    model = GeoLB_VisionTransformer(**model_args)
    return model


@register_model
def dofa_vit_so400m_patch14_siglip_384(pretrained=False, **kwargs):
    model_args = dict(
        img_size=384, patch_size=14, embed_dim=1152, depth=27, num_heads=16, mlp_ratio=3.7362, class_token=False, global_pool='map',
    )
    model = GeoLB_VisionTransformer(**model_args)
    return model

class TimmModel(nn.Module):
    """ timm model adapter
    """

    def __init__(
            self,
            model_name,
            embed_dim,
            image_size=224,
            pool='avg',
            proj='linear',
            proj_bias=False,
            drop=0.,
            drop_path=None,
            patch_drop=None,
            pretrained=False,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError("Please `pip install timm` to use timm models.")
        self.image_size = to_2tuple(image_size)
        self.DOFA = False

        # setup kwargs that may not be common across all models
        timm_kwargs = {}
        if drop_path is not None:
            timm_kwargs['drop_path_rate'] = drop_path
        if patch_drop is not None:
            timm_kwargs['patch_drop_rate'] = patch_drop

        custom_pool = pool in ('abs_attn', 'rot_attn')
        if proj:
            assert proj in ("linear", "mlp", "none")
        extra_proj = proj in ("linear", "mlp")
        if not extra_proj and not custom_pool:
            # use network classifier head as projection if no proj specified and no custom pooling used
            # if projection is explicitly set to "none" will be pass through from network trunk
            proj_dim = 0 if proj == 'none' else embed_dim
            if "dofa" in model_name:
                self.trunk = timm.create_model(model_name, pretrained=True)
                self.DOFA = True
            else:
                self.trunk = timm.create_model(
                    model_name,
                    num_classes=proj_dim,
                    global_pool=pool,
                    pretrained=pretrained,
                    **timm_kwargs,
                )
            prev_chs = embed_dim
        else:
            self.trunk = timm.create_model(
                model_name,
                pretrained=pretrained,
                **timm_kwargs,
            )
            feat_size = self.trunk.default_cfg.get('pool_size', None)
            feature_ndim = 1 if not feat_size else 2
            if custom_pool:
                assert feature_ndim == 2
                # if attn pooling used, remove both classifier and default pool
                self.trunk.reset_classifier(0, global_pool='')
            else:
                # reset global pool if pool config set, otherwise leave as network default
                reset_kwargs = dict(global_pool=pool) if pool else {}
                self.trunk.reset_classifier(0, **reset_kwargs)
            prev_chs = self.trunk.num_features

        head_layers = OrderedDict()

        # Add custom pooling to head
        if pool == 'abs_attn':
            head_layers['pool'] = AbsAttentionPool2d(prev_chs, feat_size=feat_size, out_features=embed_dim)
            prev_chs = embed_dim
        elif pool == 'rot_attn':
            head_layers['pool'] = RotAttentionPool2d(prev_chs, out_features=embed_dim)
            prev_chs = embed_dim

        # NOTE attention pool ends with a projection layer, so proj should usually be set to '' if such pooling is used
        if proj == 'linear':
            head_layers['drop'] = nn.Dropout(drop)
            head_layers['proj'] = nn.Linear(prev_chs, embed_dim, bias=proj_bias)
        elif proj == 'mlp':
            head_layers['mlp'] = Mlp(prev_chs, 2 * embed_dim, embed_dim, drop=(drop, 0), bias=(True, proj_bias))

        self.head = nn.Sequential(head_layers)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        """ lock modules
        Args:
            unlocked_groups (int): leave last n layer groups unlocked (default: 0)
        """
        if not unlocked_groups:
            # lock full model
            for param in self.trunk.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self.trunk)
        else:
            # NOTE: partial freeze requires latest timm (master) branch and is subject to change
            try:
                # FIXME import here until API stable and in an official release
                from timm.models.helpers import group_parameters, group_modules
            except ImportError:
                raise RuntimeError(
                    'Please install latest timm `pip install git+https://github.com/rwightman/pytorch-image-models`')
            matcher = self.trunk.group_matcher()
            gparams = group_parameters(self.trunk, matcher)
            max_layer_id = max(gparams.keys())
            max_layer_id = max_layer_id - unlocked_groups
            for group_idx in range(max_layer_id + 1):
                group = gparams[group_idx]
                for param in group:
                    self.trunk.get_parameter(param).requires_grad = False
            if freeze_bn_stats:
                gmodules = group_modules(self.trunk, matcher, reverse=True)
                gmodules = {k for k, v in gmodules.items() if v <= max_layer_id}
                freeze_batch_norm_2d(self.trunk, gmodules)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        try:
            self.trunk.set_grad_checkpointing(enable)
        except Exception as e:
            logging.warning('grad checkpointing not supported for this timm image tower, continuing without...')

    def forward(self, x, wvs=None):
        if self.DOFA:
            x, sfeats = self.trunk(x, wvs)
        else:
            x = self.trunk(x)
        x = self.head(x)
        #logger.debug(sfeats.shape)
        if self.DOFA:
            return x, sfeats
        else:
            return x