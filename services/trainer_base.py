import torch
from transformers import AutoTokenizer
from utils.custom_logger import logger
from utils.data_loader import DataLoader
from services.training_service.trainer import Trainer, NERPOSModel
from services.data_preprocess.preprocessor import Preprocessor
from services.data_preprocess.data_splitter import DataSplitter
from services.data_preprocess.dataset import BanglaDataset
from configs.data_config import GlobalDataConfig, ModelConfig

class TrainerBase:
    def __init__(self):
        self.file_path = GlobalDataConfig.RAW_DATA_DIR
        self.max_len = ModelConfig.MAX_LEN
        self.batch_size = ModelConfig.BATCH_SIZE
        self.epochs = ModelConfig.EPOCHS
        self.test_size = ModelConfig.TEST_SIZE
        self.Tokenizer = AutoTokenizer.from_pretrained(GlobalDataConfig.PRETRAINED_MODEL_NAME)
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = ModelConfig.RANDOM_STATE
        self.save_model_path = GlobalDataConfig.MODEL_SAVE_PATH
        self.onnx_model_path = GlobalDataConfig.ONNX_MODEL_PATH
        self.opset_version = ModelConfig.OPSET_VERSION

    def load_and_preprocess_data(self):
        data_loader = DataLoader()
        df = data_loader.load_data()
        preprocessor = Preprocessor()
        df = preprocessor.normalize_tokens(df)
        pos_tag_to_id, ner_tag_to_id = preprocessor.create_mappings(df)
        texts, labels = preprocessor.prepare_dataset(df, pos_tag_to_id, ner_tag_to_id)

        return texts, labels, pos_tag_to_id, ner_tag_to_id

    def split_data(self, texts, labels):
        splitter = DataSplitter(test_size=self.test_size, random_state=self.random_state)
        train_texts, val_texts, train_labels, val_labels = splitter.split(texts, labels)
        logger.success("Data splitted successfully")
        return train_texts, val_texts, train_labels, val_labels

    def prepare_datasets(self, train_texts, val_texts, train_labels, val_labels):
        train_dataset = BanglaDataset(train_texts, train_labels, self.Tokenizer, max_len=self.max_len)
        val_dataset = BanglaDataset(val_texts, val_labels, self.Tokenizer, max_len=self.max_len)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        logger.success("Datasets prepared successfully")

    def initialize_model(self, num_pos_tags, num_ner_tags):
        self.model = NERPOSModel(num_pos_tags=num_pos_tags, num_ner_tags=num_ner_tags).to(self.device)
        logger.info("Model initialized successfully using {}".format(self.device))

    def train_model(self):
        trainer = Trainer(self.model, self.train_loader, self.val_loader, self.device)
        trainer.train(epochs=self.epochs)
        logger.success("Training completed successfully")

    def save_model(self):
        torch.save(self.model.state_dict(), self.save_model_path)
        logger.success("Model saved successfully to {}".format(self.save_model_path))

    def export_to_onnx(self):
        # Switch the model to evaluation mode
        self.model.eval()

        # Get a real batch from the validation loader
        batch = next(iter(self.val_loader))

        # Use this batch as input for the ONNX export
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        # Export the model to ONNX
        torch.onnx.export(
            self.model,  # The trained model
            (input_ids, attention_mask),  # The real input data batch
            self.onnx_model_path,  # Path where the ONNX model will be saved
            export_params=True,  # Store the trained parameter weights inside the model file
            opset_version=self.opset_version,  # The ONNX version to export the model to
            input_names=['input_ids', 'attention_mask'],  # The model's input names
            output_names=['pos_logits', 'ner_logits'],  # The model's output names
            dynamic_axes={'input_ids': {0: 'batch_size'},  # Allow variable batch size
                          'attention_mask': {0: 'batch_size'},
                          'pos_logits': {0: 'batch_size'},
                          'ner_logits': {0: 'batch_size'}}
        )

        logger.success(f"Model exported to ONNX format at {self.onnx_model_path}")

    def run(self):
        texts, labels, pos_tag_to_id, ner_tag_to_id = self.load_and_preprocess_data()
        train_texts, val_texts, train_labels, val_labels = self.split_data(texts, labels)
        self.prepare_datasets(train_texts, val_texts, train_labels, val_labels)
        self.initialize_model(num_pos_tags=len(pos_tag_to_id), num_ner_tags=len(ner_tag_to_id))
        self.train_model()
        self.save_model()
        self.export_to_onnx()

