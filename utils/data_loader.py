import pandas as pd
from configs.logger_configs import LoggerMessage
from utils.custom_logger import logger
from configs.data_config import GlobalDataConfig


class DataLoader:
    def __init__(self):
        self.raw_data_dir = GlobalDataConfig.RAW_DATA_DIR

    def load_data(self):
        with open(self.raw_data_dir, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        data = {'sentence': [], 'token': [], 'pos_tag': [], 'ner_tag': []}
        current_sentence = None

        for line in lines:
            if line.strip() and not line.startswith('token') and '\t' not in line:
                current_sentence = line.strip()
            elif line.strip() and line.count('\t') == 2:
                token, pos_tag, ner_tag = line.strip().split('\t')
                data['sentence'].append(current_sentence)
                data['token'].append(token)
                data['pos_tag'].append(pos_tag)
                data['ner_tag'].append(ner_tag)

        df = pd.DataFrame(data)
        logger.success(LoggerMessage.DATA_LOAD_DONE)
        return df

