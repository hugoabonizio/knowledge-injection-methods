## Setup

```console
pip install -r requirements.txt
```

If you want to run training, install these additional requirements:

```console
pip install wheel  # Optional, might be required to install Flash Attention
pip install flash-attn --no-build-isolation
pip install liger-kernel trl sentencepiece protobuf
```

## Structure

This repository contains:

- `augmentation/`: scripts to generate synthetic augmentations
- `custom_tasks/`: custom tasks for LM Eval Harness modified to use RAG within the pipeline
- `exp/`: scripts to run the train and evaluation pipeline
- `outputs/`: generated outputs, including answers, evaluations, and control set runs
- `plots/`: scripts to generate plots from data in the `outputs/` folder
- `scripts/`: scripts to train, generate answers and evaluate the models

## LM Eval Harness results reproduction

### Control datasets

```console
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=bfloat16 \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --batch_size 16 \
    --trust_remote_code \
    --output_path ./outputs/control/
```

```console
python scripts/read-lm-eval-results.py outputs/control/meta-llama__Llama-2-7b-chat-hf/
```

### RAG

#### Control datasets + RAG

```console
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=bfloat16 \
    --include_path ./custom_tasks \
    --tasks openbookqa_ragtop1,arc_easy_ragtop1,winogrande_ragtop1,hellaswag_ragtop1,arc_challenge_ragtop1,piqa_ragtop1,boolq_ragtop1 \
    --batch_size 16 \
    --output_path ./outputs/control_rag/top1/
```

```console
python scripts/read-lm-eval-results.py outputs/control_rag/top1/meta-llama__Llama-2-7b-chat-hf/
```

```console
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=bfloat16 \
    --include_path ./custom_tasks \
    --tasks openbookqa_ragtop5,arc_easy_ragtop5,winogrande_ragtop5,hellaswag_ragtop5,arc_challenge_ragtop5,piqa_ragtop5,boolq_ragtop5 \
    --batch_size 16 \
    --output_path ./outputs/control_rag/top5/
```

```console
python scripts/read-lm-eval-results.py outputs/control_rag/top5/meta-llama__Llama-2-7b-chat-hf/
```
