import logging
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fairseq.data.data_utils import compute_mask_indices
from fairseq.distributed import fsdp_wrap
from fairseq.models import register_model, BaseFairseqModel
from fairseq.models.data2vec import Data2VecAudioConfig, get_annealed_rate
from fairseq.models.transformer import Embedding
from fairseq.models.wav2vec import pad_to_multiple, TransformerSentenceEncoderLayer, make_conv_pos
from fairseq.modules import LayerNorm, PositionalEmbedding, SamePad, TransposeLast, EMAModuleConfig, EMAModule
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.conformer_layer import ConformerWav2Vec2EncoderLayer
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.tasks.speech_code_pretraining import SpeechCodePretrainingTask, SpeechCodePretrainingConfig
from fairseq.utils import index_put

logger = logging.getLogger(__name__)


@dataclass
class Data2VecCodeConfig(Data2VecAudioConfig):
    no_scale_embedding: bool = field(
        default=False,
        metadata={"help": "not scale embedding"},
    )
    no_sin_pos_embed: bool = field(
        default=False,
        metadata={"help": "not sinusoidal positional embedding"},
    )
    learned_pos: bool = field(
        default=False,
        metadata={"help": "whether the sin pos embed is leanred"}
    )
    no_pos_conv: bool = field(
        default=False,
        metadata={"help": "not positional convolution"},
    )
    code_mask: bool = field(
        default=False,
        metadata={"help": "whether to apply mask according to code boundary."
                          "by default, apply span mask."}
    )


@register_model("data2vec_code", dataclass=Data2VecCodeConfig)
class Data2VecCodeModel(BaseFairseqModel):
    def __init__(self, cfg: Data2VecCodeConfig, task_cfg: SpeechCodePretrainingConfig, source_dict):
        super().__init__()

        self.cfg = cfg

        # ema required
        self.ema = None
        self.embed = cfg.encoder_embed_dim

        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale

        # build code embedding
        self.encoder_embed_tokens = self.build_embedding(
            source_dict, self.embed
        )
        self.padding_idx = self.encoder_embed_tokens.padding_idx

        # apply static sin position embedding
        if not cfg.no_sin_pos_embed:
            self.embed_positions = PositionalEmbedding(
                int(task_cfg.max_sample_size / 320) + 1,
                self.embed,
                self.padding_idx,
                learned=cfg.learned_pos,
            )
            logger.info(f"Use sin pos embedding.")
        else:
            self.embed_positions = None
            logger.info(f"Will NOT use sin pos embedding.")

        self.embed_scale = 1.0 if cfg.no_scale_embedding \
            else math.sqrt(self.embed)

        # mask related configs. will not mask by channel
        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        # dropout related
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        # use self-defined encoder
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.final_proj = nn.Linear(self.embed, self.embed)

        self.num_updates = 0

        self.code_mask = cfg.code_mask
        logger.info(f"apply_code_mask={self.code_mask}")

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        return Embedding(num_embeddings, embed_dim, padding_idx)

    # copy from data2vec_audio
    def make_ema_teacher(self):
        ema_config = EMAModuleConfig(
            ema_decay=self.cfg.ema_decay,
            ema_fp32=True,
        )
        skip_keys = set()
        if self.cfg.ema_layers_only:
            self.cfg.ema_transformer_only = True
            if self.encoder.pos_conv is not None:
                for k, _ in self.encoder.pos_conv.named_parameters():
                    skip_keys.add(f"pos_conv.{k}")

        self.ema = EMAModule(
            self.encoder if self.cfg.ema_transformer_only else self,
            ema_config,
            skip_keys=skip_keys,
        )

    # copy from data2vec_audio
    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.ema is None and self.final_proj is not None:
            logger.info(f"making ema teacher")
            self.make_ema_teacher()
        elif self.training and self.ema is not None:
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.encoder if self.cfg.ema_transformer_only else self)

        self.num_updates = num_updates

    # copy from data2vec_audio
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state

    # copy from data2vec_audio
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self.ema is not None:
            k = prefix + "_ema"
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def build_model(cls, cfg: Data2VecCodeConfig, task: SpeechCodePretrainingTask):
        return cls(cfg, task.cfg, task.source_dictionary)

    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
        mask_channel_indices=None,
    ):
        """
        basically copy from data2vec audio. but remove channel masking.
        """
        B, T, C = x.shape

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=1,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                    require_same_masks=self.cfg.require_same_masks,
                    mask_dropout=self.cfg.mask_dropout,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
        else:
            mask_indices = None

        return x, mask_indices

    def apply_code_mask(
            self,
            code: torch.Tensor,
            features: torch.Tensor,
            mask_prob: float,
            padding_mask=None
    ):
        """
        Apply mask by code boundary to the feature.
        Args:
            code: the uncompressed source code
            features: the code embedding to be masked
            mask_prob: mask probability, the only parameter to control the mask
            padding_mask: same shape as code. False -> no padding; True -> padding.

        Returns:
            masked_features, mask_indices
        """
        code = code.cpu()
        bsz = code.shape[0]

        all_masks = []

        for i in range(bsz):
            compress_code, counts = torch.unique_consecutive(code[i], return_counts=True)
            if padding_mask is None:
                has_pad = False
            else:
                has_pad = padding_mask[i].any()
            sample_size = (compress_code.shape[0] - 1) if has_pad else compress_code.shape[0]
            mask_num = int(sample_size * mask_prob + np.random.rand())

            mask_idx = np.random.choice(sample_size, size=mask_num, replace=False)
            mask_idx = np.sort(mask_idx)

            sample_mask = np.full(shape=(compress_code.shape[0],), fill_value=False)
            sample_mask[mask_idx] = True

            sample_mask = np.reshape(sample_mask, newshape=(-1, 1))

            uncompress_sample_mask = np.repeat(sample_mask, counts)

            all_masks.append(uncompress_sample_mask)

        mask_indices = torch.from_numpy(np.stack(all_masks)).to(features.device)
        features = index_put(features, mask_indices, self.mask_emb)
        return features, mask_indices

    def forward_embedding(self, src_tokens):
        token_embedding = self.encoder_embed_tokens(src_tokens)
        x = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = x + self.embed_positions(src_tokens)
        return x

    def forward(
            self,
            source_codes,
            source=None,
            padding_mask=None,
            mask=True,
            features_only=False,
            layer=None,
            mask_indices=None,
            mask_channel_indices=None,
            padding_count=None,
    ):
        # will not be used here
        source = None
        # code -> embedding
        encoder_padding_mask = source_codes.eq(self.padding_idx)
        has_pads = encoder_padding_mask.any()
        # B,T,C
        features = self.forward_embedding(src_tokens=source_codes)

        if has_pads:
            features = features * (1 - encoder_padding_mask.unsqueeze(-1).type_as(features))
        # B * T * C
        features = self.layer_norm(features)

        # compare code index with pad index to get padding mask
        if encoder_padding_mask is not None and encoder_padding_mask.any():
            padding_mask = encoder_padding_mask
        else:
            padding_mask = None

        pre_encoder_features = features.clone()
        features = self.dropout_input(features)

        if mask:
            if self.code_mask:
                x, mask_indices = self.apply_code_mask(
                    source_codes,
                    features,
                    self.mask_prob,
                    padding_mask
                )
            else:
                x, mask_indices = self.apply_mask(
                    features,
                    padding_mask,
                    mask_indices=mask_indices,
                    mask_channel_indices=mask_channel_indices,
                )
        else:
            x = features
            mask_indices = None

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=layer,
        )

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "layer_results": layer_results,
            }

        result = {
            "losses": {},
        }

        with torch.no_grad():
            self.ema.model.eval()

            # the only option is ema transformer
            y, layer_results = self.ema.model.extract_features(
                pre_encoder_features,
                padding_mask=padding_mask,
                min_layer=self.cfg.encoder_layers - self.average_top_k_layers,
            )
            y = {
                "x": y,
                "padding_mask": padding_mask,
                "layer_results": layer_results,
            }

            # the rest are copied from super class
            target_layer_results = [l[2] for l in y["layer_results"]]

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    tl.permute(1, 2, 0) for tl in target_layer_results  # TBC -> BCT
                ]
                permuted = True

            if self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in target_layer_results
                ]

            if self.cfg.instance_norm_target_layer:
                target_layer_results = [
                    F.instance_norm(tl.float()) for tl in target_layer_results
                ]

            if permuted:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
                ]

            if self.cfg.group_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-2:])
                    for tl in target_layer_results
                ]

            if self.cfg.layer_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:])
                    for tl in target_layer_results
                ]

            y = sum(target_layer_results) / len(target_layer_results)

            if self.cfg.layer_norm_targets:
                y = F.layer_norm(y.float(), y.shape[-1:])

            if self.cfg.instance_norm_targets:
                y = F.instance_norm(y.float().transpose(1, 2)).transpose(1, 2)

            if not permuted:
                y = y.transpose(0, 1)

            y = y[mask_indices]

        x = x[mask_indices]
        x = self.final_proj(x)

        sz = x.size(-1)

        if self.loss_beta == 0:
            loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(
                x.float(), y.float(), reduction="none", beta=self.loss_beta
            ).sum(dim=-1)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(sz)

        result["losses"]["regression"] = loss.sum() * scale

        if "sample_size" not in result:
            result["sample_size"] = loss.numel()

        with torch.no_grad():
            result["target_var"] = self.compute_var(y)
            result["pred_var"] = self.compute_var(x.float())

        if self.num_updates > 5000 and result["target_var"] < self.cfg.min_target_var:
            logger.error(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )
            raise Exception(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )
        if self.num_updates > 5000 and result["pred_var"] < self.cfg.min_pred_var:
            logger.error(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )
            raise Exception(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )

        if self.ema is not None:
            result["ema_decay"] = self.ema.get_decay() * 1000

        # add log for mask token number
        result["total_token_num"] = torch.sum((~padding_mask).int()) if padding_mask is not None \
                                    else source_codes.numel()

        return result

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y ** 2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs ** 2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()

    def extract_features(
        self, source, padding_mask, mask=False, layer=None
    ):
        res = self.forward(
            source_codes=source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            layer=layer,
        )
        return res

    def remove_pretraining_modules(self, last_layer=None):
        self.final_proj = None
        self.ema = None
        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )


class TransformerEncoder(nn.Module):
    def build_encoder_layer(self, args: Data2VecCodeConfig):
        if args.layer_type == "transformer":
            layer = TransformerSentenceEncoderLayer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=self.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                activation_fn=args.activation_fn,
                layer_norm_first=args.layer_norm_first,
            )
        elif args.layer_type == "conformer":
            layer = ConformerWav2Vec2EncoderLayer(
                embed_dim=self.embedding_dim,
                ffn_embed_dim=args.encoder_ffn_embed_dim,
                attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,
                activation_fn="swish",
                attn_type=args.attn_type,
                use_fp16=args.fp16,
                pos_enc_type="abs",
            )
        layer = fsdp_wrap(layer)
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        return layer

    def __init__(self, args: Data2VecCodeConfig):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.required_seq_len_multiple = args.required_seq_len_multiple

        if args.no_pos_conv:
            self.pos_conv = None
            logger.info(f"No pos_conv is used.")
        else:
            logger.info(f"pos_conv is used.")
            pos_conv_depth = getattr(args, "pos_conv_depth", 1)
            if pos_conv_depth > 1:
                num_layers = args.pos_conv_depth
                k = max(3, args.conv_pos // num_layers)

                def make_conv_block(e, k, g, l):
                    return nn.Sequential(
                        *[
                            nn.Sequential(
                                nn.Conv1d(
                                    e,
                                    e,
                                    kernel_size=k,
                                    padding=k // 2,
                                    groups=g,
                                ),
                                SamePad(k),
                                TransposeLast(),
                                LayerNorm(e, elementwise_affine=False),
                                TransposeLast(),
                                nn.GELU(),
                            )
                            for _ in range(l)
                        ]
                    )

                self.pos_conv = make_conv_block(
                    self.embedding_dim, k, args.conv_pos_groups, num_layers
                )

            else:
                self.pos_conv = make_conv_pos(
                    self.embedding_dim,
                    args.conv_pos,
                    args.conv_pos_groups,
                )

        self.layers = nn.ModuleList(
            [self.build_encoder_layer(args) for _ in range(args.encoder_layers)]
        )
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(
            self,
            x,
            padding_mask=None,
            tgt_layer=None,
            min_layer=0,
    ):

        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        if self.pos_conv is not None:
            x_conv = self.pos_conv(x.transpose(1, 2))
            x_conv = x_conv.transpose(1, 2)
            x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        # pad to the sequence length dimension
        x, pad_length = pad_to_multiple(
            x, self.required_seq_len_multiple, dim=-2, value=0
        )
        if pad_length > 0 and padding_mask is None:
            padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            padding_mask[:, -pad_length:] = True
        else:
            padding_mask, _ = pad_to_multiple(
                padding_mask, self.required_seq_len_multiple, dim=-1, value=True
            )
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random() if self.layerdrop > 0 else 1
            if not self.training or (dropout_probability > self.layerdrop):
                x, (z, lr) = layer(
                    x, self_attn_padding_mask=padding_mask, need_weights=False
                )
                if i >= min_layer:
                    layer_results.append((x, z, lr))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # undo paddding
        if pad_length > 0:
            x = x[:, :-pad_length]

            def undo_pad(a, b, c):
                return (
                    a[:-pad_length],
                    b[:-pad_length] if b is not None else b,
                    c[:-pad_length],
                )

            layer_results = [undo_pad(*u) for u in layer_results]

        return x, layer_results

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict
