from torch.utils.data import Dataset
import os
import json
"""
This implementation assumes that the test data is stored in a directory with one JSON file per programming exercise.
Each JSON file should contain a dictionary with the following keys:
- problem: a string representing the problem description
- test_case: a string representing the test case
- solution: a string representing the code solution
"""
class ProgrammingExerciseDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.exercise_files = os.listdir(data_dir)

    def __len__(self):
        return len(self.exercise_files)

    def __getitem__(self, idx):
        exercise_file = os.path.join(self.data_dir, self.exercise_files[idx])
        with open(exercise_file, 'r') as f:
            exercise_data = json.load(f)

        problem = exercise_data['problem']
        test_case = exercise_data['test_case']
        solution = exercise_data['solution']

        return problem, test_case, solution
