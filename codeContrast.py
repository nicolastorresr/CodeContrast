import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from problem_description_encoder import ProblemDescriptionEncoder
from test_case_encoder import TestCaseEncoder
from code_solution_encoder import CodeSolutionEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CodeContrast(nn.Module):
    def __init__(self, config):
        super(CodeContrast, self).__init__()
        self.problem_encoder = ProblemDescriptionEncoder(config.bert_path, config.hidden_size)
        self.test_case_encoder = TestCaseEncoder(config.input_size, config.output_size, config.hidden_size)
        self.code_encoder = CodeSolutionEncoder(config.roberta_path, config.hidden_size)
        self.projection = nn.Linear(config.hidden_size * 3, config.hidden_size)

        self.problem_encoder = self.problem_encoder.to(device)
        self.test_case_encoder = self.test_case_encoder.to(device)
        self.code_encoder = self.code_encoder.to(device)
        self.projection = self.projection.to(device)

    def forward(self, problem_input, test_case_input, code_input):
        problem_repr = self.problem_encoder(problem_input['input_ids'], problem_input['attention_mask'])
        test_case_repr = self.test_case_encoder(test_case_input['input_seq'], test_case_input['output_seq'])
        code_repr = self.code_encoder(code_input['input_ids'], code_input['attention_mask'])

        combined_repr = torch.cat((problem_repr, test_case_repr, code_repr), dim=1)
        projected_repr = self.projection(combined_repr)
        return projected_repr

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

def train(model, train_loader, optimizer, epoch, config):
    model.train()
    loss_accumulator = 0.0

    for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
        problem_input = {
            'input_ids': batch['problem_input_ids'].to(device),
            'attention_mask': batch['problem_attention_mask'].to(device)
        }
        test_case_input = {
            'input_seq': batch['test_case_input_seq'].to(device),
            'output_seq': batch['test_case_output_seq'].to(device)
        }
        code_input = {
            'input_ids': batch['code_input_ids'].to(device),
            'attention_mask': batch['code_attention_mask'].to(device)
        }

        positive_pairs = []
        negative_pairs = []

        positive_repr = model(problem_input, test_case_input, code_input)
        positive_pairs.append(positive_repr)

        for _ in range(config.num_negatives):
            negative_problem_input = {
                'input_ids': batch['negative_problem_input_ids'].to(device),
                'attention_mask': batch['negative_problem_attention_mask'].to(device)
            }
            negative_test_case_input = {
                'input_seq': batch['negative_test_case_input_seq'].to(device),
                'output_seq': batch['negative_test_case_output_seq'].to(device)
            }
            negative_code_input = {
                'input_ids': batch['negative_code_input_ids'].to(device),
                'attention_mask': batch['negative_code_attention_mask'].to(device)
            }

            negative_repr = model(negative_problem_input, negative_test_case_input, negative_code_input)
            negative_pairs.append(negative_repr)

        loss = contrastive_loss(positive_pairs, negative_pairs, temperature=config.temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_accumulator += loss.item()

    return loss_accumulator / len(train_loader)

# Training loop
model = CodeContrast(config)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

for epoch in range(config.num_epochs):
    train_loss = train(model, train_loader, optimizer, epoch, config)
    print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}')

    # Save model checkpoints, evaluate on validation set, etc.
