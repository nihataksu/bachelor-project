export NO_PLT_SHOW=True

RANDOM_SEEDS=("32")
PATCH_SIZES=("4")


for param1 in "${RANDOM_SEEDS[@]}"; do
    for param2 in "${PATCH_SIZES[@]}"; do
        export RANDOM_SEED=$param1
        export PATCH_SIZE=$param2
        python experiment.py
    done
done