import torch
import torch.onnx
from configs.data_config import ModelConfig, GlobalDataConfig
from services.training_service.trainer import NERPOSModel
from utils.custom_logger import logger


class ONNXExporter:
    def __init__(self, model, val_loader, device, onnx_model_path="ner_pos_model.onnx"):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.onnx_model_path = GlobalDataConfig.ONNX_MODEL_PATH
        self.opset_version = ModelConfig.ONNX_OPSET_VERSION
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
