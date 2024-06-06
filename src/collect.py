import os
import json
from hivit.result_collector import find_folders_with_parameters
from hivit.result_collector import read_test_results
from hivit.parameters import Parameters





def collect_results(base_directory):
    #base_directory = '/Users/commander/repos/github.com/nihataksu/bachelor-project/src/experiment_results/seed_size'
    folders = find_folders_with_parameters(base_directory)

    parameters_list=[];
    for folder in folders:
        parameters_file = os.path.join(folder,"parameters.json")
        paramers = Parameters()
        paramers.load_from_json(parameters_file)
        
        learning_result_file = os.path.join(folder,"learned_test_result.json")
        if os.path.isfile(learning_result_file):
            paramers.MODEL_LEARNING_RESULT_LOSS,paramers.MODEL_LEARNING_RESULT_ACCURACY= read_test_results(learning_result_file)

        hilbert_result_file = os.path.join(folder,"hilbert_test_result.json")
        if os.path.isfile(hilbert_result_file):
            paramers.MODEL_HILBERT_RESULT_LOSS,paramers.MODEL_HILBERT_RESULT_ACCURACY= read_test_results(hilbert_result_file)
        
        noembeding_result_file = os.path.join(folder,"no_positional_embedding_test_result.json")
        if os.path.isfile(noembeding_result_file):
            paramers.MODEL_NOEMBEDING_RESULT_LOSS,paramers.MODEL_NOEMBEDING_RESULT_ACCURACY= read_test_results(noembeding_result_file)
        
        parameters_list.append(paramers.dictionary())
    return parameters_list
    
#print(paramers.BATCH_SIZE,paramers.MODEL_HILBERT_RESULT_LOSS,paramers.MODEL_HILBERT_RESULT_ACCURACY)


list=collect_results('/Users/commander/nihat_experiments/2024-06-06-11-27')
result = {}
result["results"]=list
with open("2024-06-06-11-27.json", "w") as file:
    json.dump(result, file, indent=4)

#print(result)
