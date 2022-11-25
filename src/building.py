"""This code is for integrated post-processing"""

from logging import Logger
from argparse import Namespace

from .shape.shaper import Shaper
from .much.muching import MuchAV
from .utils import shape_from_extractor_args, batching


class CEJC_Builder:
    def __init__(self, logger: Logger, args: Namespace):
        self.logger = logger
        self.args = args

        self.shaper = Shaper(logger)
        self.marger = MuchAV(logger)

        self.batch_size = self.shaper.batch_size

    def __call__(self, shaper_list: list, muchav_list: list):
        shaper_list = batching(shaper_list, self.batch_size)
        muchav_list = batching(muchav_list, self.batch_size)

        for batch in shaper_list:
            # _path_list = shape_from_extractor_args(batch)
            shape_result = self.shaper(batch)

            sh_path = []
            shape_dicts = []
            norms = []
            for step_result in shape_result:
                sh_path.append(step_result[0])
                shape_dicts.append(step_result[1])
                norms.append(step_result[2])

    def get_shape_inputs(self, path_list):
        _path_list = shape_from_extractor_args(path_list)
        return _path_list

    # def get_muchav_inputs(self, )
