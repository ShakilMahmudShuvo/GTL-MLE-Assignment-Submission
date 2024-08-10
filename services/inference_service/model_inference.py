import torch
from transformers import AutoTokenizer
from bnunicodenormalizer import Normalizer
from configs.data_config import ModelConfig, GlobalDataConfig
from services.training_service.trainer import NERPOSModel

class InferenceService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = GlobalDataConfig.MODEL_SAVE_PATH
        self.max_len = ModelConfig.MAX_LEN
        self.pos_tag_to_id = None
        self.ner_tag_to_id = None
        self.id_to_pos_tag = None
        self.id_to_ner_tag = None

        # Load the model
        self.model = self.load_model()
        self.model.eval()

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(GlobalDataConfig.PRETRAINED_MODEL_NAME)

        # Initialize the normalizer
        self.normalizer = Normalizer()

    def load_model(self):
        # Create a fresh instance of the model with the same architecture
        model = NERPOSModel(num_pos_tags=len(self.pos_tag_to_id), num_ner_tags=len(self.ner_tag_to_id))
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        return model

    def set_tag_mappings(self, pos_tag_to_id, ner_tag_to_id):
        self.pos_tag_to_id = pos_tag_to_id
        self.ner_tag_to_id = ner_tag_to_id
        self.id_to_pos_tag = {idx: tag for tag, idx in pos_tag_to_id.items()}
        self.id_to_ner_tag = {idx: tag for tag, idx in ner_tag_to_id.items()}

    def infer(self, sentence):
        # Normalize the sentence
        sentence_tokens = sentence.split()  # Basic tokenization
        normalized_tokens = [self.normalizer(token)['normalized'] for token in sentence_tokens]

        # Tokenize and encode the input sentence
        inputs = self.tokenizer(normalized_tokens, is_split_into_words=True, return_tensors="pt",
                                padding='max_length', truncation=True, max_length=self.max_len)

        # Move tensors to the appropriate device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Get the model predictions
        with torch.no_grad():
            pos_logits, ner_logits = self.model(input_ids, attention_mask)

        # Get the predicted tags
        pos_preds = torch.argmax(pos_logits, dim=-1).cpu().numpy()[0]
        ner_preds = torch.argmax(ner_logits, dim=-1).cpu().numpy()[0]

        # Decode the predictions
        pos_tags = [self.id_to_pos_tag[idx] for idx in pos_preds if idx != ModelConfig.IGNORE_INDEX]
        ner_tags = [self.id_to_ner_tag[idx] for idx in ner_preds if idx != ModelConfig.IGNORE_INDEX]

        # Trim to the length of the input sentence
        token_count = len(sentence_tokens)
        pos_tags = pos_tags[:token_count]
        ner_tags = ner_tags[:token_count]

        # Combine tokens with their predicted POS and NER tags
        results = list(zip(sentence_tokens, pos_tags, ner_tags))

        return results