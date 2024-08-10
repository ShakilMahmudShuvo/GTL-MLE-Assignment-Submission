import torch
from torch.utils.data import Dataset

class BanglaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        tokens = self.texts[index]
        labels = self.labels[index]

        tokens = [token for token in tokens if token is not None]

        if isinstance(tokens, list) and all(isinstance(token, str) for token in tokens):
            encoded = self.tokenizer(tokens, is_split_into_words=True, padding='max_length',
                                     truncation=True, max_length=self.max_len, return_attention_mask=True)

            word_ids = encoded.word_ids()
            pos_ids = [-100] * len(encoded['input_ids'])
            ner_ids = [-100] * len(encoded['input_ids'])

            for i, word_id in enumerate(word_ids):
                if word_id is not None and word_id < len(labels['pos_tag']):
                    pos_ids[i] = labels['pos_tag'][word_id]
                    ner_ids[i] = labels['ner_tag'][word_id]

            return {
                'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
                'pos_tag_ids': torch.tensor(pos_ids, dtype=torch.long),
                'ner_tag_ids': torch.tensor(ner_ids, dtype=torch.long)
            }
        else:
            raise ValueError("tokens should be a list of strings")

    def __len__(self):
        return len(self.texts)
