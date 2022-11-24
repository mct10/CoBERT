# CoBERT
## Pre-training
code_teacher_1:
```
fairseq-hydra-train -m \
    --config-dir cobert/config/code_teacher_1/pretraining \
    --config-name base_librispeech \
    task.data=/path/to/manifest \
    task.label_dir=/path/to/codes \
    model.label_rate=50 \
    dataset.valid_subset=dev_other \
    dataset.train_subset=train \
    common.user_dir=/path/to/CoBERT/cobert/
```
code_teacher_2:
```
fairseq-hydra-train -m \
    --config-dir cobert/config/code_teacher_2/pretraining \
    --config-name base_librispeech \
    task.data=/path/to/manifest \
    dataset.valid_subset=dev_other \
    dataset.train_subset=train \
    +model.no_sin_pos_embed=true \
    common.user_dir=/path/to/CoBERT/cobert/
```
cobert:
```
fairseq-hydra-train -m \
    --config-dir cobert/config/cobert/pretraining \
    --config-name base_librispeech \
    task.data=/path/to/manifest \
    dataset.valid_subset=dev_other \
    dataset.train_subset=train \
    model.code_teacher_ckpt=/path/to/teacher model \
    common.user_dir=/users/cmeng9/CoBERT/cobert/
```

## Fine-tuning
code_teacher_1:
```
fairseq-hydra-train -m \
  --config-dir cobert/config/code_teacher_1/finetuning \
  --config-name base_100h \
  task.data=/path/to/manifest \
  task.label_dir=/path/to/label \
  +task.code_dir=/path/to/codes \
  model.w2v_path=/path/to/ckpt \
  common.user_dir=/path/to/CoBERT/cobert/
```
code_teacher_2:
```
fairseq-hydra-train \
    --config-dir cobert/config/code_teacher_2/finetuning \
    --config-name base_100h \
    task.data=/path/to/manifest \
    task.label_dir=/path/to/label \
    model.w2v_path=/path/to/ckpt \
    dataset.train_subset=train_100h \
    dataset.valid_subset=dev_other \
    common.user_dir=/path/to/CoBERT/cobert/
```
cobert:
```
fairseq-hydra-train \
    --config-dir fairseq/examples/wav2vec/config/finetuning \
    --config-name base_100h \
    task.data=/path/to/manifest \
    model.w2v_path=/path/to/ckpt \
    +model.normalize=true \
    dataset.train_subset=train_100h \
    dataset.valid_subset=dev_other \
    common.user_dir=/path/to/CoBERT/cobert/
```
## Infer
code_teacher_1:
```
```
code_teacher_2:
```
```
cobert:
```
```