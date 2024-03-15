import torch
from codeContrast import CodeContrast
from config import Config
from dataset import ProgrammingExerciseDataset

config = Config()

# Load trained model
model = CodeContrast(config)
model.load_state_dict(torch.load('path/to/model/checkpoint.pth'))
model.eval()

# Load test dataset
test_dataset = ProgrammingExerciseDataset('path/to/test/data')

# Generate programming exercises
generated_exercises = []
for problem, test_case, solution in test_dataset:
    with torch.no_grad():
        problem_input = model.problem_encoder(problem)
        test_case_input = model.test_case_encoder(test_case)
        generated_solution = model.code_decoder(problem_input, test_case_input)
    generated_exercises.append((problem, test_case, generated_solution))

# Save generated exercises to file
with open('generated_exercises.txt', 'w') as f:
    for exercise in generated_exercises:
        f.write(f"Problem: {exercise[0]}\n")
        f.write(f"Test Case: {exercise[1]}\n")
        f.write(f"Solution: {exercise[2]}\n\n")