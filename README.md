
## Generating Parameters

- Edit generate_parameters.py 
- Name the experiment
- choose an output folder
- edit the parameters for the experiments you want to run 
- Run generate_parameters.py to generate individual experiment files
- the experiments are created under the experiments folder

```python
experiment_name = "cifar10"
output_folder = "data/experiments"
os.makedirs(output_folder, exist_ok=True)

# Example usage:
overrides = {
    "DATASET_NAME": ["cifar10"],
    "EPOCHS": ["200"],
    "RANDOM_SEEDS": ["32"],
    "PATCH_SIZES": ["4"],
    "BATCH_SIZES": ["128"],
    "HIDDEN_DIMS": ["768"],
    "NUM_ENCODERSS": ["8", "6"],
    "NUM_HEADSS": ["8", "6"],
}
```

```bash
# generate experiment parameters
python generate_parameters.py
```

## Training 

- There is a folder trainig queue and under that there are 4 subfolders 
    - **queue**: stores the experiment files to be processed
    - **lock**: stores locked files   
    - **reults**: for each experiment there are folders consisting of the results of the experiments 
    - **processed**: consist of experiment files that are done

- experiment2.sh gets the files in the queue and processed them and for every file it process it creates a lock 


- Local training

    -  default location of the trainig folder is src/training


- Cluster training

    - defaults location of the trainig folder is /scratch/$USER/training


- put the experiment files to queue folder in training folder
























