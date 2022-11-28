# CoBERT
## Install
```
git clone https://github.com/mct10/CoBERT.git
cd CoBERT
git submodule update --init fairseq
pip install --editable fairseq/
cd fairseq
python setup.py build develop
cd ..
```
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
    model.code_teacher_ckpt=/path/to/teacher \
    model.code_teacher_type=code_teacher_2 \
    +model.multi_outputs=true \
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
python cobert/infer.py \
  --config-dir cobert/config/code_teacher_1/decode \
  --config-name infer_viterbi \
  task.data=/path/to/manifest \
  task.normalize=false \
  +task.code_dir=/path/to/codes \
  common_eval.path=/path/to/ckpt \
  dataset.gen_subset=dev_other \
  common.user_dir=/path/to/CoBERT/cobert/
```
code_teacher_2:
```
python cobert/infer.py \
  --config-dir cobert/config/code_teacher_2/decode \
  --config-name infer_viterbi \
  task.data=/path/to/manifest \
  task.normalize=false \
  task.label_dir=/path/to/label \
  common_eval.path=/path/to/ckpt \
  dataset.gen_subset=dev_other \
  common.user_dir=/path/to/CoBERT/cobert/
```
cobert:
```
python /users/cmeng9/CoBERT/cobert/infer.py \
  --config-dir fairseq/examples/speech_recognition/new/conf \
  --config-name infer \
  task=audio_finetuning \
  task.labels=ltr \
  decoding.type=viterbi \
  task.data=/path/to/manifest \
  task.normalize=false \
  common_eval.path=/path/to/ckpt \
  dataset.gen_subset=dev_other \
  common.user_dir=/path/to/CoBERT/cobert/
```