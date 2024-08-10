import os

class GlobalDataConfig:
    # Ensure necessary directories exist
    directories = [
        "data/saved_models",
        "data/onnx_saved_models"
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Now you can safely use these directories in your code
    RAW_DATA_DIR = "data/raw_data/data.tsv"
    PRETRAINED_MODEL_NAME = 'csebuetnlp/banglabert'
    MODEL_SAVE_PATH = "data/saved_models/ner_pos_model.pth"
    ONNX_MODEL_PATH = "data/onnx_saved_models/ner_pos_model.onnx"


class ModelConfig:
    IGNORE_INDEX = -100
    LEARNING_RATE = 5e-5
    MAX_LEN = 128
    BATCH_SIZE = 128
    EPOCHS = 10
    RANDOM_STATE = 42
    TEST_SIZE = 0.1
    OPSET_VERSION = 12
