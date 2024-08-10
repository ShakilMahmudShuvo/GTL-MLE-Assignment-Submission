import logging
from configs.logger_configs import LOGGING_CONFIG

class Color(object):
    """
     utility to return ansi colored text.
    """

    COLORS = LoggerInfo.COLORS

    PREFIX = LoggerInfo.PREFIX
    SUFFIX = LoggerInfo.SUFFIX

    def get_colored_text(self, text, color=None):
        color = self.COLORS.get(color, 37)
        return f"{self.PREFIX}{color}m{text}{self.SUFFIX}"


class ColorFormatter(logging.Formatter):
    MAPPED_LEVEL = LoggerInfo.MAPPED_LEVEL
    DEFAULT_COLOR = LoggerInfo.DEFAULT_COLOR

    # DATE_TIME_FORMATTER = '%y-%m-%d %H:%M:%S.%sf'

    def format_level_name(self, level):
        return ('{: <7}'.format(level))

    def format_date_time(self, date_time):
        return date_time[2:].replace(',', '.')

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()

        # Format date time
        if self.usesTime():
            record.asctime = self.format_date_time(self.formatTime(record, self.datefmt))

        # Format error message
        if record.levelname == "ERROR":
            message = f"{message}\n{record.relativeCreated}{record.stack_info}"

        # Format level name
        record.levelname = self.format_level_name(record.levelname)

        complete_msg = f"\u200B{record.asctime} [{record.levelname}] [recommendation-engine, {record.threadName}, , , ] {record.module} - {message}"
        return complete_msg


# Adding Custom Level: success level
logging.SUCCESS = 25  # between WARNING and INFO
logging.addLevelName(logging.SUCCESS, 'SUCCESS')

formatter = ColorFormatter("%(asctime)s - %(module)s - %(funcName)s - line:%(lineno)d - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)

setattr(logger, 'success', lambda message, *args: logger._log(logging.SUCCESS, message, args))

logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)