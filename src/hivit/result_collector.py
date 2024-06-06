import os
import json

def read_test_results(file_path):
    """
    Reads a JSON file and returns a tuple containing the test_loss and test_accuracy.

    Parameters:
    file_path (str): The path to the JSON file.

    Returns:
    tuple: A tuple containing the test_loss and test_accuracy.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    test_loss = data.get("test_loss")
    test_accuracy = data.get("test_accuracy")
    
    return test_loss, test_accuracy

def find_folders_with_parameters(base_directory):
    folders_with_parameters = []
    
    for root, dirs, files in os.walk(base_directory):
        if 'parameters.json' in files:
            folders_with_parameters.append(root)
    
    return folders_with_parameters