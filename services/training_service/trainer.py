import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel
from utils.custom_logger import logger
from configs.data_config import GlobalDataConfig, ModelConfig

class NERPOSModel(nn.Module):
    def __init__(self, num_pos_tags, num_ner_tags):
        super(NERPOSModel, self).__init__()
        self.bert = AutoModel.from_pretrained(GlobalDataConfig.PRETRAINED_MODEL_NAME)
        self.pos_classifier = nn.Linear(self.bert.config.hidden_size, num_pos_tags)
        self.ner_classifier = nn.Linear(self.bert.config.hidden_size, num_ner_tags)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pos_logits = self.pos_classifier(sequence_output)
        ner_logits = self.ner_classifier(sequence_output)
        return pos_logits, ner_logits


class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=ModelConfig.IGNORE_INDEX)
        self.optimizer = optim.Adam(self.model.parameters(), lr=ModelConfig.LEARNING_RATE)

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                pos_tag_ids = batch['pos_tag_ids'].to(self.device)
                ner_tag_ids = batch['ner_tag_ids'].to(self.device)

                self.optimizer.zero_grad()
                pos_logits, ner_logits = self.model(input_ids, attention_mask)
                loss_pos = self.criterion(pos_logits.view(-1, len(pos_tag_ids)), pos_tag_ids.view(-1))
                loss_ner = self.criterion(ner_logits.view(-1, len(ner_tag_ids)), ner_tag_ids.view(-1))
                total_loss = loss_pos + loss_ner
                total_loss.backward()
                self.optimizer.step()

            logger.info(f"Epoch {epoch + 1}, Loss: {total_loss.item()}")
