export TOKENIZERS_PARALLELISM=false

MODEL_ID="meta-llama/Llama-2-7b-chat-hf"

API_KEY=""

N_GPUS=$(nvidia-smi -L | wc -l)
TOTAL_BATCH_SIZE=8
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACCUMULATION=$(( TOTAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * N_GPUS) ))

for i in 1; do
    for N_REPEATS in 1 5 10 20 40; do
        for TEMPERATURE in 10; do
            for GENERATOR in gpt4o llama2; do
                MODEL_NAME="llama2_7b_chat_paraphrases_repeat${N_REPEATS}-temperature_${TEMPERATURE}-${GENERATOR}"
                OUTPUT_PATH="outputs/models/${MODEL_NAME}"

                echo $MODEL_NAME
                if [ ! -d "${OUTPUT_PATH}" ]; then
                    torchrun --nnodes=1 --nproc-per-node=${N_GPUS} scripts/train.py \
                        --model_name_or_path "${MODEL_ID}" \
                        --dataset_name "hugo/knowledge-injection-1" \
                        --dataset_config_name "paraphrases-repeat_${N_REPEATS}-temperature_${TEMPERATURE}-${GENERATOR}" \
                        --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
                        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
                        --do_train \
                        --num_train_epochs 2 \
                        --learning_rate 5e-5 \
                        --weight_decay 0.1 \
                        --bf16 \
                        --lr_scheduler_type cosine \
                        --gradient_checkpointing \
                        --optim adamw_torch_fused \
                        --save_only_model \
                        --overwrite_output_dir \
                        --save_strategy no \
                        --logging_strategy steps \
                        --logging_steps 2 \
                        --warmup_ratio 0.5 \
                        --use_liger_kernel \
                        --report_to none \
                        --seed ${i} \
                        --max_seq_length 4096 \
                        --output_dir "${OUTPUT_PATH}"
                fi

                GENERATIONS_PATH="outputs/results/${MODEL_NAME}_closedbook_generations.json"
                EVALUATIONS_PATH="outputs/results/${MODEL_NAME}_closedbook_evaluations.json"

                if [ ! -f "$GENERATIONS_PATH" ]; then
                    python scripts/generate-answers.py \
                        --model_name "${OUTPUT_PATH}" \
                        --output "${GENERATIONS_PATH}" \
                        --batch_size 64
                fi

                if [ ! -f "$EVALUATIONS_PATH" ]; then
                    python scripts/evaluate-answers.py \
                        --generations "${GENERATIONS_PATH}" \
                        --output "${EVALUATIONS_PATH}" \
                        --batch_size 10 \
                        --api_key "${API_KEY}"
                fi

                LMEVAL_RESULTS="outputs/control/outputs__models__${MODEL_NAME}"
                if [ ! -d ${LMEVAL_RESULTS} ]; then
                    accelerate launch -m lm_eval --model hf \
                        --model_args pretrained=${OUTPUT_PATH},dtype=bfloat16 \
                        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
                        --batch_size 8 \
                        --output_path "outputs/control/"
                    python scripts/read-lm-eval-results.py "${LMEVAL_RESULTS}"
                fi


                # rm -f ${OUTPUT_PATH}/model* ${OUTPUT_PATH}/tokenizer*
            done
        done
    done
done
