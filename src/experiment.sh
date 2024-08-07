#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=experiment
#SBATCH --time=2-00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu

if [ -z "$SLURM_JOB_ID" ]
then
    echo "not habrok"
else 
    echo "habrok"
    # Purge
    module purge
    # Set up Python module
    module load Python/3.9.6-GCCcore-11.2.0
    # For Conda Env
    module load Anaconda3/2023.09-0
    # conda env activate
    conda activate bachelor-project && echo "activated conda bachelor-project"
    export WORKING_ROOT="/scratch/$USER/experiment_results"
    export DATASET_ROOT="/scratch/$USER/data"
fi 


export NO_PLT_SHOW=True


EXPERIMENT_BEGIN=$(date +"%Y-%m-%d %H:%M:%S")
echo "EXPERIMENT BEGIN:" $EXPERIMENT_BEGIN


export RANDOM_SEED=42
export BATCH_SIZE=256
export EPOCHS=100
export PATIENCE=200
export LEARNING_RATE="1e-3"
export LEARNING_RATE_SCHEDULER_NAME=""
export PATCH_SIZE=4
export NUM_HEADS=12
export DROPOUT=0.3
export HIDDEN_DIM=512
export ADAM_WEIGHT_DECAY=0
export ADAM_BETAS="(0.9, 0.999)"
export ACTIVATION="gelu"
export NUM_ENCODERS=6

export TRANSFORM_RANDOM_HORIZONTAL_FLIP_ENABLED="False"
export TRANSFORM_RANDOM_ROTATION_ENABLED="False"
export TRANSFORM_RANDOM_ROTATION_DEGREE="0"
export TRANSFORM_RANDOM_CROP_ENABLED="False"
export TRANSFORM_RANDOM_CROP_PADDING="4"
export TRANSFORM_COLOR_JITTER_ENABLED="False"
export TRANSFORM_COLOR_JITTER_BRIGHTNESS="0.2"
export TRANSFORM_COLOR_JITTER_CONTRAST="0.2"
export TRANSFORM_COLOR_JITTER_SATURATION="0.2"
export TRANSFORM_COLOR_JITTER_HUE="0.2"
export TRANSFORM_RANDOM_ERASING="False"

export EXECUTE_MODEL_LEARNING="True"
export EXECUTE_MODEL_HILBERT="False"
export EXECUTE_MODEL_NOEMBEDING="False"
export EXECUTE_MODEL_RESNET_EMBEDING="False"
export EXECUTE_MODEL_RESNET_HILBERT_EMBEDING="False"

export TRAINER_BOT_TOKEN="7453991407:AAHxY8z6n8N4r84e_fYK9XQIhmLfbplNTck"
export TRAINER_BOT_CHAT_ID="-4236180801"

export EPOCHS=200
export DATASET_NAME="food101"


RANDOM_SEEDS=("32")
PATCH_SIZES=("16")
BATCH_SIZES=("128")
HIDDEN_DIMS=("768")
NUM_ENCODERSS=("8")
NUM_HEADSS=("8")

for param1 in "${RANDOM_SEEDS[@]}"; do
    for param2 in "${HIDDEN_DIMS[@]}"; do
        for param3 in "${PATCH_SIZES[@]}"; do
            for param4 in "${NUM_ENCODERSS[@]}"; do
                for param5 in "${NUM_HEADSS[@]}"; do
                    for param6 in "${BATCH_SIZES[@]}"; do
                        export RANDOM_SEED=$param1
                        export HIDDEN_DIM=$param2
                        export PATCH_SIZE=$param3
                        export NUM_ENCODERS=$param4
                        export NUM_HEADS=$param5
                        export BATCH_SIZE=$param6
                        python experiment.py
                    done
                done
            done 
        done
    done 
done

EXPERIMENT_END=$(date +"%Y-%m-%d %H:%M:%S")


echo "EXPERIMENT BEGIN:" $EXPERIMENT_BEGIN
echo "EXPERIMENT END:" $EXPERIMENT_END