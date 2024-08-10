import onnxruntime as ort
import numpy as np
import torch
from transformers import AutoTokenizer
from bnunicodenormalizer import Normalizer
from configs.data_config import ModelConfig, GlobalDataConfig
from utils.data_loader import DataLoader
from services.data_preprocess.preprocessor import Preprocessor

class ONNXInferenceService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.onnx_model_path = GlobalDataConfig.ONNX_MODEL_PATH
        self.max_len = ModelConfig.MAX_LEN

        # Load data and extract tag mappings
        self.pos_tag_to_id, self.ner_tag_to_id = self.load_and_prepare_mappings()

        self.id_to_pos_tag = {idx: tag for tag, idx in self.pos_tag_to_id.items()}
        self.id_to_ner_tag = {idx: tag for tag, idx in self.ner_tag_to_id.items()}

        # Load the ONNX model
        self.ort_session = ort.InferenceSession(self.onnx_model_path)

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(GlobalDataConfig.PRETRAINED_MODEL_NAME)

        # Initialize the normalizer
        self.normalizer = Normalizer()

    def load_and_prepare_mappings(self):
        data_loader = DataLoader()
        df = data_loader.load_data()

        preprocessor = Preprocessor()
        df = preprocessor.normalize_tokens(df)
        pos_tag_to_id, ner_tag_to_id = preprocessor.create_mappings(df)

        return pos_tag_to_id, ner_tag_to_id

    def infer(self, sentence):
        # Normalize the sentence
        sentence_tokens = sentence.split()  # Basic tokenization
        normalized_tokens = [self.normalizer(token)['normalized'] for token in sentence_tokens]

        # Tokenize and encode the input sentence
        inputs = self.tokenizer(normalized_tokens, is_split_into_words=True, return_tensors="np",
                                padding='max_length', truncation=True, max_length=self.max_len)

        # Extract the input_ids and attention_mask directly as NumPy arrays
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Perform inference using ONNX model
        ort_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        ort_outs = self.ort_session.run(None, ort_inputs)

        # Get the outputs
        pos_logits, ner_logits = ort_outs

        # Get the predicted tags
        pos_preds = np.argmax(pos_logits, axis=-1)[0]
        ner_preds = np.argmax(ner_logits, axis=-1)[0]

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
