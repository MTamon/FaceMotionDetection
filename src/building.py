"""This code is for integrated post-processing"""

from logging import Logger

from .shape.shaper import Shaper
from .much.muching import MuchAV
from .utils import shape_from_extractor_args, batching


class CEJC_Builder:
    def __init__(self, logger: Logger):
        self.logger = logger

        self.shaper = Shaper(logger)
        self.marger = MuchAV(logger)

        self.batch_size = self.shaper.batch_size

    def __call__(self, path_list: list):
        path_list = batching(path_list)

        for batch in path_list:
            _path_list = shape_from_extractor_args(batch)
            shape_result = self.shaper(_path_list)

            sh_path = []
            shape_dicts = []
            norms = []
            for step_result in shape_result:
                sh_path.append(step_result[0])
                shape_dicts.append(step_result[1])
                norms.append(step_result[2])
