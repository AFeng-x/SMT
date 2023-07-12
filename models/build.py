from .swin_transformer import SwinTransformer
from .smt import SMT


def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm


    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                norm_layer=layernorm,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=config.FUSED_WINDOW_PROCESS)

    elif model_type == 'smt':
        model = SMT(img_size=config.DATA.IMG_SIZE,
                    in_chans=config.MODEL.SMT.IN_CHANS,
                    num_classes=config.MODEL.NUM_CLASSES,
                    embed_dims=config.MODEL.SMT.EMBED_DIMS,
                    ca_num_heads=config.MODEL.SMT.CA_NUM_HEADS,
                    sa_num_heads=config.MODEL.SMT.SA_NUM_HEADS,
                    mlp_ratios=config.MODEL.SMT.MLP_RATIOS,
                    qkv_bias=config.MODEL.SMT.QKV_BIAS,
                    qk_scale=config.MODEL.SMT.QK_SCALE,
                    use_layerscale=config.MODEL.SMT.USE_LAYERSCALE,
                    depths=config.MODEL.SMT.DEPTHS,
                    ca_attentions=config.MODEL.SMT.CA_ATTENTIONS,
                    head_conv=config.MODEL.SMT.HEAD_CONV,
                    expand_ratio=config.MODEL.SMT.EXPAND_RATIO,
                    drop_rate=config.MODEL.DROP_RATE,
                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                    use_checkpoint=config.TRAIN.USE_CHECKPOINT)   

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
