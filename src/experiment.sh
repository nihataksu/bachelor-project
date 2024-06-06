export NO_PLT_SHOW=True


EXPERIMENT_BEGIN=$(date +"%Y-%m-%d %H:%M:%S")
echo "EXPERIMENT BEGIN:" $EXPERIMENT_BEGIN


export RANDOM_SEED=42
export BATCH_SIZE=256
export EPOCHS=200
export PATIENCE=200
export LEARNING_RATE="1e-3"
export LEARNING_RATE_SCHEDULER_NAME="CosineAnnealingLR"
export NUM_CLASSES=101
export PATCH_SIZE=8
export IMAGE_SIZE=128
export IN_CHANNELS=3
export NUM_HEADS=12
export DROPOUT=0.1
export HIDDEN_DIM=512
export ADAM_WEIGHT_DECAY=0
export ADAM_BETAS="(0.9, 0.999)"
export ACTIVATION="gelu"
export NUM_ENCODERS=6

export EXECUTE_MODEL_LEARNING="True"
export EXECUTE_MODEL_HILBERT="True"
export EXECUTE_MODEL_NOEMBEDING="False"

export TRAINER_BOT_TOKEN="7453991407:AAHxY8z6n8N4r84e_fYK9XQIhmLfbplNTck"
export TRAINER_BOT_CHAT_ID="-4236180801"

export EPOCHS=300
export DATASET_NAME="food101"


RANDOM_SEEDS=("32")
PATCH_SIZES=("8")
BATCH_SIZES=("128")
HIDDEN_DIMS=("1024")
NUM_ENCODERSS=("12")
NUM_HEADSS=("12")

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