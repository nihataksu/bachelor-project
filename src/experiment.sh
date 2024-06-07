export NO_PLT_SHOW=True


EXPERIMENT_BEGIN=$(date +"%Y-%m-%d %H:%M:%S")
echo "EXPERIMENT BEGIN:" $EXPERIMENT_BEGIN


export RANDOM_SEED=42
export BATCH_SIZE=256
export EPOCHS=100
export PATIENCE=200
export LEARNING_RATE="1e-3"
export LEARNING_RATE_SCHEDULER_NAME="CosineAnnealingLR"
export NUM_CLASSES=10
export PATCH_SIZE=4
export IMAGE_SIZE=28
export IN_CHANNELS=1
export NUM_HEADS=12
export DROPOUT=0.001
export HIDDEN_DIM=512
export ADAM_WEIGHT_DECAY=0
export ADAM_BETAS="(0.9, 0.999)"
export ACTIVATION="gelu"
export NUM_ENCODERS=6

export EXECUTE_MODEL_LEARNING="True"
export EXECUTE_MODEL_HILBERT="True"
export EXECUTE_MODEL_NOEMBEDING="False"
export EXECUTE_MODEL_RESNET_EMBEDING="True"

export TRAINER_BOT_TOKEN="7453991407:AAHxY8z6n8N4r84e_fYK9XQIhmLfbplNTck"
export TRAINER_BOT_CHAT_ID="-4236180801"

export EPOCHS=100
export DATASET_NAME="mnist"


RANDOM_SEEDS=("32")
PATCH_SIZES=("4")
BATCH_SIZES=("512")
HIDDEN_DIMS=("768" "512")
NUM_ENCODERSS=("8" "6" "4")
NUM_HEADSS=("12" "8" "6")

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