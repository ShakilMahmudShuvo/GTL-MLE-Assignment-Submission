import torch
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
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

    def evaluate_model(self, pos_tag_to_id, ner_tag_to_id):
        id_to_pos_tag = {idx: tag for tag, idx in pos_tag_to_id.items()}
        id_to_ner_tag = {idx: tag for tag, idx in ner_tag_to_id.items()}

        metrics = self.evaluate(self.model, self.val_loader, pos_tag_to_id, ner_tag_to_id, id_to_pos_tag, id_to_ner_tag,
                                self.device)

        logger.success("Evaluation completed successfully")
        logger.info(f"POS Tagging - Accuracy: {metrics['POS Tagging']['Accuracy']:.4f}, "
                    f"Precision: {metrics['POS Tagging']['Precision']:.4f}, "
                    f"Recall: {metrics['POS Tagging']['Recall']:.4f}, "
                    f"F1 Score: {metrics['POS Tagging']['F1 Score']:.4f}")
        logger.info(f"NER Tagging - Accuracy: {metrics['NER Tagging']['Accuracy']:.4f}, "
                    f"Precision: {metrics['NER Tagging']['Precision']:.4f}, "
                    f"Recall: {metrics['NER Tagging']['Recall']:.4f}, "
                    f"F1 Score: {metrics['NER Tagging']['F1 Score']:.4f}")

        return metrics

    @staticmethod
    def evaluate(model, dataloader, pos_tag_to_id, ner_tag_to_id, id_to_pos_tag, id_to_ner_tag, device):
        model.eval()

        all_pos_preds = []
        all_pos_labels = []
        all_ner_preds = []
        all_ner_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pos_tag_ids = batch['pos_tag_ids'].to(device)
                ner_tag_ids = batch['ner_tag_ids'].to(device)

                pos_logits, ner_logits = model(input_ids, attention_mask)

                pos_preds = torch.argmax(pos_logits, dim=-1).cpu().numpy()
                ner_preds = torch.argmax(ner_logits, dim=-1).cpu().numpy()

                pos_tag_ids = pos_tag_ids.cpu().numpy()
                ner_tag_ids = ner_tag_ids.cpu().numpy()

                # Flatten the predictions and labels, and remove padding (ignore_index = -100)
                for i in range(pos_preds.shape[0]):
                    pos_pred_flat = pos_preds[i][pos_tag_ids[i] != -100]
                    pos_label_flat = pos_tag_ids[i][pos_tag_ids[i] != -100]
                    all_pos_preds.extend(pos_pred_flat)
                    all_pos_labels.extend(pos_label_flat)

                    ner_pred_flat = ner_preds[i][ner_tag_ids[i] != -100]
                    ner_label_flat = ner_tag_ids[i][ner_tag_ids[i] != -100]
                    all_ner_preds.extend(ner_pred_flat)
                    all_ner_labels.extend(ner_label_flat)

        # Calculate POS tag metrics
        pos_precision, pos_recall, pos_f1, _ = precision_recall_fscore_support(
            all_pos_labels, all_pos_preds, average='macro', labels=np.unique(all_pos_labels)
        )
        pos_accuracy = accuracy_score(all_pos_labels, all_pos_preds)

        # Calculate NER tag metrics
        ner_precision, ner_recall, ner_f1, _ = precision_recall_fscore_support(
            all_ner_labels, all_ner_preds, average='macro', labels=np.unique(all_ner_labels)
        )
        ner_accuracy = accuracy_score(all_ner_labels, all_ner_preds)

        metrics = {
            "POS Tagging": {
                "Accuracy": pos_accuracy,
                "Precision": pos_precision,
                "Recall": pos_recall,
                "F1 Score": pos_f1
            },
            "NER Tagging": {
                "Accuracy": ner_accuracy,
                "Precision": ner_precision,
                "Recall": ner_recall,
                "F1 Score": ner_f1
            }
        }

        return metrics

    def run(self):
        texts, labels, pos_tag_to_id, ner_tag_to_id = self.load_and_preprocess_data()
        train_texts, val_texts, train_labels, val_labels = self.split_data(texts, labels)
        self.prepare_datasets(train_texts, val_texts, train_labels, val_labels)
        self.initialize_model(num_pos_tags=len(pos_tag_to_id), num_ner_tags=len(ner_tag_to_id))
        self.train_model()
        self.save_model()
        self.export_to_onnx()
        metrics = self.evaluate_model(pos_tag_to_id, ner_tag_to_id)
        return metrics
