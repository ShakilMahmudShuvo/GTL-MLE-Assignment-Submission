class LoggerInfo:
    DEFAULT_COLOR = 'white'
    PREFIX = '\033['
    SUFFIX = '\033[0m'
    COLORS = {
        'black': 30,
        'red': 31,
        'green': 32,
        'yellow': 33,
        'blue': 34,
        'magenta': 35,
        'cyan': 36,
        'white': 37,
        'bgred': 41,
        'bggrey': 100
    }

    MAPPED_LEVEL = {
        'INFO': 'blue',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bgred',
        'DEBUG': 'cyan',
        'SUCCESS': 'green'
    }


class LoggerMessage:
    DATA_LOAD_DONE = "All data load has done."
    GLOBAL_DATA_READ_DONE = "Reading Global Data Files are done."
    MODEL_TRAINING_STARTED = "Model training has been started..."
    MODEL_TRAINING_DONE = "Model training have done."
    MODEL_SAVING_DONE = "Model saving to the storage have done."
    DATA_SAVING_DONE = "Data saving to the storage have done."
    PROCESSING_STARTED = "Processing Started... .."
    PROCESSING_DONE = "Processing Done!"