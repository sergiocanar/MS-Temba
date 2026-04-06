"""
models_cvs_temba.py

Registers the CVSTemba model variant with timm.

CVSTemba is MSTemba with:
  - in_feat_dim = 1024  (EVA02 global image embeddings)
  - num_classes = 3     (CVS criteria c1, c2, c3)

All other architecture details (hierarchical dilated SSMs, diversity loss,
block auxiliary heads) are inherited unchanged from MSTemba.
"""

from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from models_MSTemba import MSTemba


@register_model
def cvstemba(pretrained: bool = False, **kwargs):
    kwargs.setdefault("in_feat_dim", 1024)
    kwargs.setdefault("num_classes", 3)

    model = MSTemba(
        embed_dims=[256, 384, 576],
        depths=[1, 1, 1],
        d_state=16,
        **kwargs,
    )
    model.default_cfg = _cfg()

    if pretrained:
        raise NotImplementedError("No pretrained CVSTemba checkpoint available yet.")

    return model
