import logging


def configure_logger(logger, verbose: bool):
    logger.setLevel(logging.DEBUG)

    # create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('logfile.log')
    c_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    # create formatters and add them to handlers
    c_format = logging.Formatter('%(levelname)s: %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

