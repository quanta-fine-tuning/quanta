#!/bin/bash
MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

TASK=${TASK:-SIQATest} # please download the dataset from https://github.com/AGI-Edgerunners/LLM-Adapters

EPOCH=${EPOCH:-1}
BS=${BS:-4}
GA=1
LR=${LR:-5e-5}
SEED=${SEED:-31415}
TRAIN_SET_SEED=${TRAIN_SET_SEED:-271828}
MODE=${MODE:-quanta}
TYPE=$MODE
DEVICE=${DEVICE:-0}
TRAINER=${TRAINER:-regular}
export CUDA_VISIBLE_DEVICES=$DEVICE
EXTRA_ARGS=""

TAG=$MODE-$EPOCH-$BS-$LR-$SEED

SAVE_INTERVAL=25000

TASK_ARGS=""

python eval.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir 'result/'$TASK-${MODEL_NAME}-$TAG \
    --tag $TAG \
    --tuning_type $TYPE \
    --train_set_seed $TRAIN_SET_SEED \
    --seed $SEED \
    --num_train 0 \
    --num_dev 0 \
    --num_eval 100000 \
    --logging_steps 10 \
    --trainer $TRAINER \
    --learning_rate $LR \
    --num_train_epochs 0 \
    --per_device_train_batch_size $BS \
    --load_best_model_at_end \
    --evaluation_strategy steps \
    --eval_step $(expr $SAVE_INTERVAL / $BS / $GA) \
    --save_strategy steps \
    --save_step $(expr $SAVE_INTERVAL / $BS / $GA) \
    --save_total_limit 2 \
    --train_as_classification \
    --target_modules "q_proj" "v_proj" \
    --quanta_per_dim_features 16 8 8 4 \
    --checkpoint_dir 'result/COMMONSENSE170K'-${MODEL_NAME}-$TAG \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"