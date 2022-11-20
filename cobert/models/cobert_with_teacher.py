import logging
import math
import os
from argparse import Namespace

from dataclasses import dataclass, field
from typing import List

from omegaconf import MISSING, II

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models.data2vec.data2vec_audio import Data2VecAudioConfig, Data2VecAudioModel
from fairseq.models.data2vec.data2vec_code import Data2VecCodeModel
from fairseq.models import register_model
from fairseq.models.hubert import HubertModel
from fairseq.models.roberta.model import RobertaModel
from fairseq.modules import (
    GradMultiply,
)
from fairseq.tasks.masked_lm import MaskedLMTask

logger = logging.getLogger(__name__)

LENGTH_TOLERANCE=4


@dataclass
class Data2VecAudioCodeConfig(Data2VecAudioConfig):
    code_teacher_ckpt: str = field(
        default=MISSING,
        metadata={"help": "The path to the ckpt of the teacher model. "
                          "If not provides, will act the same as origin data2vec."}
    )
    code_teacher_type: str = field(
        default="roberta",
        metadata={"help": "the type of code teacher"}
    )
    code_teacher_min_layer: int = field(
        default=0,
        metadata={"help": "inclusive. The first layer index whose output counts as teacher."}
    )
    code_teacher_max_layer: int = field(
        default=-1,
        metadata={"help": "exclusive. The last layer index whose output counts as teacher."}
    )
    multi_outputs: bool = field(
        default=False,
        metadata={"help": "whether to compute loss using multiplie output layers."}
    )
    code_loss_only: bool = field(
        default=False,
        metadata={"help": "whether to compute only the loss from code teacher. "
                          "if true, ema model will not be used and the training is faster."
                          "for ema loss only, use data2vec_audio directly."}
    )
    # refer to the dir defined in SpeechCodePretrainingConfig
    normalize: bool = II("task.normalize")
    code_path: str = II("task.data")


def _load_roberta(_cfg: Data2VecAudioCodeConfig):
    checkpoint = torch.load(_cfg.code_teacher_ckpt)

    import argparse

    task_args = checkpoint['cfg']['task']
    if isinstance(task_args, dict):
        task_parser = argparse.ArgumentParser()
        MaskedLMTask.add_args(task_parser)
        # 'data' field in MLM task is REQUIRED, only provide a pseudo one here
        task_args = argparse.Namespace(**{**vars(task_parser.parse_args(["PSEUDO_DATA"])), **task_args})
    assert isinstance(task_args, argparse.Namespace)
    # assign the real data path
    task_args.data = _cfg.code_path
    logger.info("task args:")
    logger.info(task_args)

    model_args = checkpoint['cfg']['model']
    if isinstance(model_args, dict):
        model_parser = argparse.ArgumentParser()
        RobertaModel.add_args(model_parser)
        model_args = argparse.Namespace(**{**vars(model_parser.parse_args([])), **model_args})
    assert isinstance(model_args, argparse.Namespace)
    logger.info("model args")
    logger.info(model_args)

    task = MaskedLMTask.setup_task(task_args)
    model = RobertaModel.build_model(model_args, task)
    model.load_state_dict(checkpoint['model'])
    return model


def _load_cobert(_cfg: Data2VecAudioCodeConfig) -> HubertModel:
    # should not override args, we need to keep the teacher as it was.
    state = checkpoint_utils.load_checkpoint_to_cpu(_cfg.code_teacher_ckpt)
    w2v_args = state.get("cfg", None)
    if w2v_args is None:
        w2v_args = convert_namespace_to_omegaconf(state["args"])

    # no need to check normalize, because code hubert does not require normalize
    # assert _cfg.normalize == w2v_args.task.normalize

    w2v_args.task.data = _cfg.code_path
    pretrain_task = tasks.setup_task(w2v_args.task)
    # This will load the stored "dictionaries" object
    pretrain_task.load_state_dict(state["task_state"])

    model = pretrain_task.build_model(w2v_args.model, from_checkpoint=True)
    # can be strict here
    model.load_state_dict(state["model"], strict=True)

    # model.remove_pretraining_modules()
    return model


def _load_data2vec_audio(_cfg: Data2VecAudioCodeConfig) -> Data2VecAudioModel:
    # This loads both data2vec_audio model and data2vec_audio_code model.
    state = checkpoint_utils.load_checkpoint_to_cpu(_cfg.code_teacher_ckpt)
    w2v_args = state.get("cfg", None)
    if w2v_args is None:
        w2v_args = convert_namespace_to_omegaconf(state["args"])

    # have to check normalize. the teacher use the same audio as the student here.
    assert _cfg.normalize == w2v_args.task.normalize

    pretrain_task = tasks.setup_task(w2v_args.task)
    # This will load the stored "dictionaries" object
    pretrain_task.load_state_dict(state["task_state"])

    # prevent loading from wrong ckpt path
    if "code_teacher_ckpt" in w2v_args.model:
        logger.info(f"Replaced code_teacher_ckpt {w2v_args.model['code_teacher_ckpt']}...")
        w2v_args.model["code_teacher_ckpt"] = "placeholder"

    # does not need the code_teacher_proj for feature extraction
    if "multi_outputs" in w2v_args.model:
        logger.info(f"Set multi_outputs {w2v_args.model['multi_outputs']} to false")
        w2v_args.model["multi_outputs"] = False
    model = pretrain_task.build_model(w2v_args.model, from_checkpoint=True)
    # delete possible teacher models (code & ema) since not needed.
    # also because we do not provide code teacher here.
    for k in list(state["model"].keys()):
        if "code_teacher" in k or "_ema" in k:
            del state["model"][k]
    # can be strict here
    model.load_state_dict(state["model"], strict=True)

    model.remove_pretraining_modules()
    return model


def _load_data2vec_code(_cfg: Data2VecAudioCodeConfig) -> Data2VecCodeModel:
    state = checkpoint_utils.load_checkpoint_to_cpu(_cfg.code_teacher_ckpt)
    w2v_args = state.get("cfg", None)
    if w2v_args is None:
        w2v_args = convert_namespace_to_omegaconf(state["args"])

    # no need to check normalization since data2vec_code does not need audio input
    # assert _cfg.normalize == w2v_args.task.normalize

    # NOTE: be sure to put dict.km.txt at /opt/tiger/pretrain_meta
    pretrain_task = tasks.setup_task(w2v_args.task)
    pretrain_task.load_state_dict(state["task_state"])

    model = pretrain_task.build_model(w2v_args.model, from_checkpoint=True)
    # delete possible teacher models (ema) since not needed.
    for k in list(state["model"].keys()):
        if "_ema" in k:
            del state["model"][k]
    # can be strict here
    model.load_state_dict(state["model"], strict=True)

    model.remove_pretraining_modules()
    return model

def _load_hubert(_cfg: Data2VecAudioCodeConfig) -> HubertModel:
    # do code input inference here.
    state = checkpoint_utils.load_checkpoint_to_cpu(_cfg.code_teacher_ckpt)
    w2v_args = state.get("cfg", None)
    if w2v_args is None:
        w2v_args = convert_namespace_to_omegaconf(state["args"])

    # need to check normalize. The performance should be better if the normalization matches
    assert _cfg.normalize == w2v_args.task.normalize

    # do not need this. we provide data using Data2VecAudioCodeModel.
    # w2v_args.task.data = _cfg.code_path
    pretrain_task = tasks.setup_task(w2v_args.task)
    # This will load the stored "dictionaries" object
    pretrain_task.load_state_dict(state["task_state"])

    model = pretrain_task.build_model(w2v_args.model, from_checkpoint=True)
    # can be strict here
    model.load_state_dict(state["model"], strict=True)

    model.remove_pretraining_modules()
    return model


@register_model("data2vec_audio_code", dataclass=Data2VecAudioCodeConfig)
class Data2VecAudioCodeModel(Data2VecAudioModel):
    cfg: Data2VecAudioCodeConfig

    def __init__(self, cfg: Data2VecAudioCodeConfig):
        super().__init__(cfg)
        # load another teacher model
        self.code_teacher_model = None
        self.code_teacher_type = cfg.code_teacher_type
        if cfg.code_teacher_ckpt is not None and os.path.exists(cfg.code_teacher_ckpt):
            logger.info(f"Will load code teacher {self.code_teacher_type} from {cfg.code_teacher_ckpt}")
            if self.code_teacher_type == "roberta":
                self.code_teacher_model: RobertaModel = _load_roberta(cfg)
            elif self.code_teacher_type == "cobert":
                self.code_teacher_model: HubertModel = _load_cobert(cfg)
            elif self.code_teacher_type == "data2vec_code":
                self.code_teacher_model: Data2VecCodeModel = _load_data2vec_code(cfg)
            elif self.code_teacher_type == "data2vec_audio":
                self.code_teacher_model = _load_data2vec_audio(cfg)
            elif self.code_teacher_type == "hubert":
                self.code_teacher_model: HubertModel = _load_hubert(cfg)
            self.code_teacher_model.requires_grad_(requires_grad=False)
            self.code_teacher_model.eval()
            # log the parameters to make sure all parameters are correctly set
            for name, param in self.named_parameters():
                logger.info(f"{name}.requires_grad={param.requires_grad}")
        else:
            logger.warning(f"Connot load code teacher from {cfg.code_teacher_ckpt}. Make sure this is fine-tuning.")

        self.multi_outputs = cfg.multi_outputs
        logger.info(f"multi-outputs={self.multi_outputs}")
        if self.multi_outputs:
            self.code_teacher_proj = nn.Linear(self.embed, self.embed)

        self.code_loss_only = cfg.code_loss_only
        logger.info(f"code_loss_only={self.code_loss_only}")

    def forward(
            self,
            source,
            source_codes=None,
            source_codes_lengths=None,
            padding_mask=None,
            mask=True,
            features_only=False,
            layer=None,
            mask_indices=None,
            mask_channel_indices=None,
            padding_count=None,
    ):
        features = source

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(features)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(features)

        features = features.transpose(1, 2)

        features = self.layer_norm(features)

        orig_padding_mask = padding_mask

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        pre_encoder_features = None
        if self.cfg.ema_transformer_only:
            pre_encoder_features = features.clone()

        features = self.dropout_input(features)

        if mask:
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
            if self.code_loss_only:
                y = None
            else:
                self.ema.model.eval()

                if self.cfg.ema_transformer_only:
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
                else:
                    y = self.ema.model.extract_features(
                        source=source,
                        padding_mask=orig_padding_mask,
                        mask=False,
                    )
                # TBC
                target_layer_results = [l[2] for l in y["layer_results"]]
                y = self._aggregate_features(target_layer_results)

            # compute code teacher representation
            # T x B x C
            code_y = None
            if self.code_teacher_type == "roberta":
                code_y = self._get_roberta_feature(source_codes)
            if self.code_teacher_type == "cobert":
                code_y, feat_padding_mask = self._get_cobert_feature(source_codes, orig_padding_mask)
                assert len(code_y) == self.cfg.code_teacher_max_layer - self.cfg.code_teacher_min_layer
                # T,B,C -> B,T,C
                # origin_hubert_feature = code_y[-1].transpose(0, 1)
            if self.code_teacher_type == "data2vec_code":
                code_y = self._get_data2vec_code_feature(source_codes)
                assert len(code_y) == self.cfg.code_teacher_max_layer - self.cfg.code_teacher_min_layer
            if self.code_teacher_type == "data2vec_audio":
                code_y = self._get_data2vec_audio_feature(source, orig_padding_mask)
                assert len(code_y) == self.cfg.code_teacher_max_layer - self.cfg.code_teacher_min_layer
            if self.code_teacher_type == "hubert":
                code_y = self._get_hubert_feature(source, orig_padding_mask)
                assert len(code_y) == self.cfg.code_teacher_max_layer - self.cfg.code_teacher_min_layer

            assert code_y is not None
            # T x B x C -> B x T x C
            code_y = self._aggregate_features(code_y)

            # possible trims here
            # do not trim for audio input teacher
            if self.code_teacher_type != "data2vec_audio":
                # trim feature to mask sure the dimensions match
                code_len = code_y.size(1)
                feature_len = mask_indices.size(1)
                if code_len < feature_len:
                    mask_indices = mask_indices[:, :code_len]
                    if abs(code_len - feature_len) > LENGTH_TOLERANCE:
                        logger.info(f"{code_len} < {feature_len}, trim y & mask")
                        logger.info(source_codes.size())
                        logger.info(mask_indices.size())

                    if not self.code_loss_only:
                        y = y[:, :code_len, :]
                        if abs(code_len - feature_len) > LENGTH_TOLERANCE:
                            logger.info(y.size())
                # trim code to match the dim of feature
                if code_len > feature_len:
                    code_y = code_y[:, :feature_len, :]
                    if abs(code_len - feature_len) > LENGTH_TOLERANCE:
                        logger.info(f"{code_len} > {feature_len}, trim code")
                        logger.info(code_y.size())

            if not self.code_loss_only:
                y = y[mask_indices]
            code_y = code_y[mask_indices]

        # trim x outside torch.no_grad(), just in case the gradient over x will be ignored
        if self.code_teacher_type != "data2vec_audio" and code_len < feature_len:
            x = x[:, :code_len, :]
            if abs(code_len - feature_len) > LENGTH_TOLERANCE:
                logger.info(f"{code_len} < {feature_len}, trim x")
                logger.info(x.size())
        x = x[mask_indices]

        if self.multi_outputs:
            # compute another projection for code teacher
            x_for_code_teacher = self.code_teacher_proj(x)

        x = self.final_proj(x)

        sz = x.size(-1)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(sz)

        if not self.code_loss_only:
            ema_loss = self._compute_loss(pred=x, target=y)
            result["losses"]["regression"] = ema_loss.sum() * scale

        if self.multi_outputs:
            code_loss = self._compute_loss(pred=x_for_code_teacher, target=code_y)
        else:
            code_loss = self._compute_loss(pred=x, target=code_y)

        result["losses"]["code_teacher"] = code_loss.sum() * scale

        if "sample_size" not in result:
            result["sample_size"] = code_loss.numel()

        with torch.no_grad():
            if not self.code_loss_only:
                result["target_var"] = self.compute_var(y)
                result["target_mean"] = self.compute_mean(y)

            result["pred_var"] = self.compute_var(x.float())
            result["code_teacher_var"] = self.compute_var(code_y)

            result["pred_mean"] = self.compute_mean(x.float())
            result["code_teacher_mean"] = self.compute_mean(code_y)

            if self.multi_outputs:
                result["pred_for_code_teacher_var"] = self.compute_var(x_for_code_teacher.float())

        # if self.code_teacher_type == "cobert":
        #     with torch.no_grad():
        #         self._compute_cobert_acc(source_codes, origin_hubert_feature, feat_padding_mask, result)

        if not self.code_loss_only:
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

        return result

    def _aggregate_features(self, target_layer_results: List[torch.FloatTensor]):
        """
        Normalize and aggregate multiple layers' outputs.
        Args:
            target_layer_results: outputs from multiple layers.

        Returns:
            a tensor
        """
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

        return y

    def _get_roberta_feature(self, source_codes):
        # use .eval() every time!!!!!
        self.code_teacher_model.eval()
        _, inner_states = self.code_teacher_model.encoder.extract_features(
            source_codes, return_all_hiddens=True)
        inner_states = inner_states["inner_states"]
        # inner_states from 1 to ignore word embedding
        code_y = inner_states[1:][self.cfg.code_teacher_min_layer:self.cfg.code_teacher_max_layer]
        return code_y

    def _get_cobert_feature(self, source_codes, padding_mask):
        # use .eval() every time!!!!!
        self.code_teacher_model: HubertModel
        self.code_teacher_model.eval()
        # T x B x C
        all_layer_results, feat_padding_mask = self.code_teacher_model.extract_features(
            source=source_codes,
            padding_mask=padding_mask,
            mask=False,
            output_layer=None,
            return_all_layers=True
        )
        all_features = [layer_result[2] for layer_result in all_layer_results]
        return all_features[self.cfg.code_teacher_min_layer:self.cfg.code_teacher_max_layer], feat_padding_mask

    def _get_data2vec_audio_feature(self, source, padding_mask):
        self.code_teacher_model: Data2VecAudioModel
        self.code_teacher_model.eval()
        ret = self.code_teacher_model.extract_features(
            source=source,
            padding_mask=padding_mask,
            mask=False,
        )
        all_layer_results = ret["layer_results"]
        all_features = [layer_result[2] for layer_result in all_layer_results]
        return all_features[self.cfg.code_teacher_min_layer:self.cfg.code_teacher_max_layer]

    def _get_data2vec_code_feature(self, source):
        self.code_teacher_model: Data2VecCodeModel
        self.code_teacher_model.eval()
        ret = self.code_teacher_model.extract_features(
            source,
            padding_mask=None,
            mask=False
        )
        all_layer_results = ret["layer_results"]
        all_features = [layer_result[2] for layer_result in all_layer_results]
        return all_features[self.cfg.code_teacher_min_layer:self.cfg.code_teacher_max_layer]

    def _get_hubert_feature(self, source, padding_mask):
        # use .eval() every time!!!!!
        self.code_teacher_model: HubertModel
        self.code_teacher_model.eval()
        # T x B x C
        all_layer_results, _ = self.code_teacher_model.extract_features(
            source=source,
            padding_mask=padding_mask,
            mask=False,
            output_layer=None,
            return_all_layers=True
        )
        all_features = [layer_result[2] for layer_result in all_layer_results]
        return all_features[self.cfg.code_teacher_min_layer:self.cfg.code_teacher_max_layer]

    def _compute_loss(self, pred, target):
        if self.loss_beta == 0:
            loss = F.mse_loss(pred.float(), target.float(), reduction="none").sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(
                pred.float(), target.float(), reduction="none", beta=self.loss_beta
            ).sum(dim=-1)
        return loss

    def _compute_cobert_acc(self, target, encoder_out, padding_mask, logging_output):
        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.code_teacher_model.target_glu:
                y = self.code_teacher_model.target_glu(y)
                negs = self.code_teacher_model.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            return self.code_teacher_model.compute_nce(proj_x, y, negs)

        target_list = [target]
        label_embs_list = \
            self.code_teacher_model.label_embs_concat.split(self.code_teacher_model.num_classes, 0)

        mask_indices = torch.full(size=encoder_out.size()[:2], fill_value=False)
        nomask_indices = torch.logical_and(~padding_mask.cpu(), ~mask_indices)
        encoder_out = encoder_out.to(self.code_teacher_model.final_proj.weight.dtype)
        proj_x_u = self.code_teacher_model.final_proj(encoder_out[nomask_indices])
        if self.code_teacher_model.untie_final_proj:
            proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
        else:
            proj_x_u_list = [proj_x_u for _ in range(len(target_list))]

        logit_u_list = [
            compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
            for i, (proj_x_u, t) in enumerate(zip(proj_x_u_list, target_list))
        ]
        fake_net_output = {"logit_u_list": logit_u_list}
        logp_u_list = self.code_teacher_model.get_logits(
            net_output=fake_net_output,
            is_masked=False
        )

        def compute_correct(logits):
            if logits.numel() == 0:
                return 0, 0
            else:
                assert logits.dim() > 1, logits.shape
                max = logits.argmax(-1) == 0
                min = logits.argmin(-1) == 0
                both = max & min
                corr = max.long().sum().item() - both.long().sum().item()
                count = max.numel()
                return corr, count

        for i, logp_u in enumerate(logp_u_list):
            corr_u, count_u = compute_correct(logp_u)
            logging_output[f"correct_u_{i}"] = corr_u
            logging_output[f"count_u_{i}"] = count_u

    @staticmethod
    def compute_mean(y):
        return y.mean()

    def extract_features(
        self, source, padding_mask, mask=False, layer=None
    ):
        res = self.forward(
            source=source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            layer=layer,
        )
        return res

    def remove_pretraining_modules(self, last_layer=None):
        super().remove_pretraining_modules(last_layer)
        self.code_teacher_model = None
        if self.multi_outputs:
            self.code_teacher_proj = None
