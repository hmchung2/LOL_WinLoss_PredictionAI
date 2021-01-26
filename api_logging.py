import logging, os
from logging import handlers
from datetime import datetime
from api_config import Config

def get_log_view(log_level, env, error_log=False, log_name='server'):

    root_path = Config.root_path

    log_form_front = '[%(levelname)s][%(filename)s:%(lineno)s][%(asctime)s]'
    set_logger = logging.getLogger(log_name)
    if log_level == 1:
        set_logger.setLevel(logging.INFO)
    else:
        set_logger.setLevel(logging.WARN)
    set_logger.handlers = []

    stream_fomatter = logging.Formatter(log_form_front + '%(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_fomatter)

    now = datetime.now()
    formatted_date = now.strftime("%Y%m%d")

    if error_log:
        log_path = os.path.join(root_path, 'error_log')
    else:
        log_path = os.path.join(root_path, 'log')
    file_handler = logging.FileHandler(os.path.join(log_path, '{}_{}.log'.format(log_name, formatted_date)))
    file_handler.setFormatter(stream_fomatter)

    set_logger.addHandler(stream_handler)
    set_logger.addHandler(file_handler)

    return set_logger
