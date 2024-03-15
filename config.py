class Config:
    def __init__(self):
        self.bert_path = 'path/to/bert-base-uncased'
        self.roberta_path = 'path/to/roberta-base'
        self.hidden_size = 768
        self.input_size = 10  # Size of input sequence for test cases
        self.output_size = 10  # Size of output sequence for test cases
        self.num_negatives = 4  # Number of negative samples per positive sample
        self.temperature = 0.1  # Temperature for contrastive loss
        self.learning_rate = 1e-5
        self.num_epochs = 10
        self.batch_size = 32
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'