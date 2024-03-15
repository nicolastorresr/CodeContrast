import torch
import torch.nn as nn
from transformers import BertModel
"""
Problem Description Encoder

This encoder uses a pretrained BERT model to encode the problem description text. 
The final hidden state corresponding to the [CLS] token is taken as the problem 
description embedding, which is then projected through a linear layer.
"""
class ProblemDescriptionEncoder(nn.Module):
    def __init__(self, bert_path, hidden_size=768):
        super(ProblemDescriptionEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]  # Take the [CLS] token representation
        projected_output = self.projection(cls_output)
        return projected_output