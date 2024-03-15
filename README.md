# CodeContrast

CodeContrast is a novel generative model that leverages contrastive learning to map programming problems, test cases, and code solutions into a shared feature space. By minimizing the distance between matching components and maximizing the distance between non-matching components, CodeContrast captures the intricate relationships between these elements, enabling the generation of coherent and aligned programming exercises.

## Features

- **Coherent Exercise Generation**: CodeContrast generates programming exercises where the problem descriptions, test cases, and code solutions are semantically aligned and coherent.
- **Diverse Exercise Generation**: The model can generate diverse programming exercises across various problem domains and programming concepts.
- **High Code Correctness**: The generated code solutions exhibit a high degree of correctness, passing a significant portion of test cases.
- **Comprehensive Test Cases**: The generated test cases provide comprehensive coverage, ensuring thorough evaluation of the generated solutions.
- **Pedagogical Value**: The generated exercises are tailored for introductory programming courses and have been validated for their pedagogical value by experts and student studies.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/nicolastorresr/CodeContrast.git
   ```

2. Install the required dependencies:

   ```bash
   cd CodeContrast
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your data: Ensure that you have a dataset of programming problems, test cases, and code solutions in the appropriate format. Refer to the `data/` directory for examples.

2. Configure the model: Update the configuration file `config.py` with your desired settings, such as the paths to pretrained models, hidden sizes, and other hyperparameters.

3. Train the model:

   ```bash
   python train.py
   ```

   This will start the training process for the CodeContrast model. You can monitor the training progress and loss values printed to the console.

4. Generate programming exercises:

   ```bash
   python generate.py
   ```

   This script will use the trained CodeContrast model to generate programming exercises, including problem descriptions, test cases, and code solutions. The generated exercises will be saved in the `generated/` directory.
