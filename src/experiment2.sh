#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --time=2-00:00
#SBATCH --mem=64000
#SBATCH --partition=regular

if [ -z "$SLURM_JOB_ID" ]
then
    export TRAINING_QUEUE_ROOT="training"
    export WORKING_ROOT="training_queue/results"
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
    export TRAINING_QUEUE_ROOT="/scratch/$USER/training"
fi 




EXPERIMENT_BEGIN=$(date +"%Y-%m-%d %H:%M:%S")
echo "EXPERIMENT BEGIN:" $EXPERIMENT_BEGIN


SOURCE_DIR="$TRAINING_QUEUE_ROOT/queue"
echo $SOURCE_DIR
PROCESSED_DIR="$TRAINING_QUEUE_ROOT/processed"
LOCK_DIR="$TRAINING_QUEUE_ROOT/lock"
LOG_FILE="experiments.log"


PROCESS_CMD="python experiment.py"


mkdir -p "$PROCESSED_DIR"
mkdir -p "$LOCK_DIR"

# Iterate over files in the source directory
for file in "$SOURCE_DIR"/*; do
    if [[ -f $file ]]; then
        # Create a lock file for the current file
        LOCK_FILE="$LOCK_DIR/$(basename "$file").lock"
        
        # Try to create the lock file atomically
        if ( set -o noclobber; echo "$$" > "$LOCK_FILE" ) 2> /dev/null; then
            trap 'rm -f "$LOCK_FILE"; exit $?' INT TERM EXIT
            
            # Source the environment variables from the file
            source "$file"
            
            # Run the process and capture the output and exit status
            $PROCESS_CMD &> "$LOG_FILE"
            PROCESS_EXIT_STATUS=$?
            
            # Check if the process ran successfully
            if [[ $PROCESS_EXIT_STATUS -eq 0 ]]; then
                # Move the file to the processed directory
                mv "$file" "$PROCESSED_DIR"
            else
                # Log the failure
                echo "Process failed for file $file. Check $LOG_FILE for details." >> "$LOG_FILE"
            fi
            
            # Remove the lock file
            rm -f "$LOCK_FILE"
            trap - INT TERM EXIT
        else
            echo "File $file is being processed by another instance."
        fi
    fi
done

EXPERIMENT_END=$(date +"%Y-%m-%d %H:%M:%S")


echo "EXPERIMENT BEGIN:" $EXPERIMENT_BEGIN
echo "EXPERIMENT END:" $EXPERIMENT_END