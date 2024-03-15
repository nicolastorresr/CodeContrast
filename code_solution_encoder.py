import torch
import torch.nn as nn
from transformers import RobertaModel
"""
Code Solution Encoder

Similar to the problem description encoder, this encoder uses a 
pretrained RoBERTa model to encode the code solution text. 
The final hidden state corresponding to the [CLS] token is taken as the code 
solution embedding, which is then projected through a linear layer.
"""
class CodeSolutionEncoder(nn.Module):
    def __init__(self, roberta_path, hidden_size=768):
        super(CodeSolutionEncoder, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_path)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]  # Take the [CLS] token representation
        projected_output = self.projection(cls_output)
        return projected_output