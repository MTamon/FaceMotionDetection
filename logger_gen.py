from logging import getLogger, StreamHandler, FileHandler, Formatter, INFO
from datetime import datetime


def set_logger(name, rootname="log/main.log"):
    dt_now = datetime.now()
    dt = dt_now.strftime('%Y%m%d_%H%M%S')
    fname = rootname + "." + dt
    logger = getLogger(name)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    handler2 = FileHandler(filename=fname)
    handler2.setFormatter(Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    handler1.setLevel(INFO)
    handler2.setLevel(INFO)  #handler2 more than Level.WARN
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(INFO)
    return logger