import torch
from torch.utils.data import DataLoader
from codeContrast import CodeContrast
from config import Config
from dataset import ProgrammingExerciseDataset
from tqdm import tqdm
import torch.nn.functional as F

# Function for training loop
def train_loop(model, data_loader, optimizer, epoch, config):
    model.train()
    loss_accumulator = 0.0

    for batch in tqdm(data_loader, desc=f'Epoch {epoch}'):
        problem_input = {
            'input_ids': batch['problem_input_ids'].to(config.device),
            'attention_mask': batch['problem_attention_mask'].to(config.device)
        }
        test_case_input = {
            'input_seq': batch['test_case_input_seq'].to(config.device),
            'output_seq': batch['test_case_output_seq'].to(config.device)
        }
        code_input = {
            'input_ids': batch['code_input_ids'].to(config.device),
            'attention_mask': batch['code_attention_mask'].to(config.device)
        }

        positive_pairs = []
        negative_pairs = []

        positive_repr = model(problem_input, test_case_input, code_input)
        positive_pairs.append(positive_repr)

        for _ in range(config.num_negatives):
            negative_problem_input = {
                'input_ids': batch['negative_problem_input_ids'].to(config.device),
                'attention_mask': batch['negative_problem_attention_mask'].to(config.device)
            }
            negative_test_case_input = {
                'input_seq': batch['negative_test_case_input_seq'].to(config.device),
                'output_seq': batch['negative_test_case_output_seq'].to(config.device)
            }
            negative_code_input = {
                'input_ids': batch['negative_code_input_ids'].to(config.device),
                'attention_mask': batch['negative_code_attention_mask'].to(config.device)
            }

            negative_repr = model(negative_problem_input, negative_test_case_input, negative_code_input)
            negative_pairs.append(negative_repr)

        loss = contrastive_loss(positive_pairs, negative_pairs, temperature=config.temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_accumulator += loss.item()

    return loss_accumulator / len(data_loader)

def contrastive_loss(positive_pairs, negative_pairs, temperature=0.1):
    positive_pairs = torch.cat(positive_pairs, dim=0)
    negative_pairs = torch.cat(negative_pairs, dim=0)

    positive_scores = torch.sum(positive_pairs.unsqueeze(1) * positive_pairs.unsqueeze(2), dim=-1) / temperature
    negative_scores = torch.sum(negative_pairs.unsqueeze(1) * positive_pairs.unsqueeze(2), dim=-1) / temperature

    positive_scores = positive_scores - positive_scores.max(dim=2, keepdim=True)[0].detach()
    negative_scores = negative_scores - negative_scores.max(dim=2, keepdim=True)[0].detach()

    positive_exp = torch.exp(positive_scores)
    negative_exp = torch.exp(negative_scores.sum(dim=2))

    positive_numer = positive_exp
    positive_denom = positive_exp + negative_exp

    loss = -torch.log(positive_numer / positive_denom).mean()
    return loss

config = Config()

# Load dataset
train_dataset = ProgrammingExerciseDataset('path/to/train/data')
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

# Initialize model
model = CodeContrast(config)
model = model.to(config.device)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
for epoch in range(config.num_epochs):
    train_loop(model, train_loader, optimizer, epoch, config)
    # Save model checkpoint