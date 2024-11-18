#!/bin/bash
MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

TASK=${TASK:-DROP}

EPOCH=${EPOCH:-3}
BS=${BS:-4}
LR=${LR:-1e-4}
SEED=${SEED:-314159}
TRAIN_SET_SEED=${TRAIN_SET_SEED:-2718281}
TRAIN=${TRAIN:-2000}
DEV=${DEV:-800}
EVAL=${EVAL:-1200}
MODE=${MODE:-quanta}
TYPE=$MODE
DEVICE=${DEVICE:-0}
TRAINER=${TRAINER:-regular}
export CUDA_VISIBLE_DEVICES=$DEVICE
EXTRA_ARGS=""

TAG=$MODE-$EPOCH-$BS-$LR-$SEED

TASK_ARGS=""

# Add additional tasks here
case $TASK in
    DROP) # You can modify the gradient accumulation steps and batch size as needed
        GA=$(expr $BS / 1)
        BS=1
        TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
        ;;
esac

python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG \
    --tag $TAG \
    --tuning_type $TYPE \
    --train_set_seed $TRAIN_SET_SEED \
    --seed $SEED \
    --num_train $TRAIN \
    --num_dev $DEV \
    --num_eval $EVAL \
    --logging_steps 10 \
    --trainer $TRAINER \
    --learning_rate $LR \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BS \
    --load_best_model_at_end \
    --evaluation_strategy steps \
    --eval_step 500 \
    --save_strategy steps \
    --save_step 500 \
    --save_total_limit 1 \
    --train_as_classification \
    --target_modules 'q_proj' 'v_proj' \
    --quanta_per_dim_features 16 8 8 4 \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"
