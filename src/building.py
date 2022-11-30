"""This code is for integrated post-processing"""

from logging import Logger
from argparse import Namespace
from typing import List

from .shape.shaper import Shaper
from .match.matching import MatchAV
from .utils import shape_from_extractor_args, batching


class CEJC_Builder:
    def __init__(self, logger: Logger, args: Namespace):
        self.logger = logger
        self.args = args

        self.shaper = Shaper(logger)
        self.merger = MatchAV(logger)

        self.batch_size = self.shaper.batch_size

    def __call__(self, shaper_list: list, matchav_list: list):
        shaper_list = batching(shaper_list, self.batch_size)
        matchav_list = batching(matchav_list, self.batch_size)

        merge_res = []

        for batch_s, batch_m in zip(shaper_list, matchav_list):
            shape_r = self.shaper(batch_s)

            # batch_m shape {".csv": path, ".wav": [path1, ...]} * batch_size
            # shape_r shape [shape_result: ndarray, norm_info, normalizer] * batch_size
            merge_res += self.merger(batch_m, shape_r)

        self.logger.info("BUILDER >> Analysis process done.")
        self.logger.info("BUILDER >> Start Optimize process.")

        prime_csv = {}
        for record in merge_res:
            if not record["__name__"] in prime_csv.keys():
                prime_csv[record["__name__"]] = []

            prime_csv[record["__name__"]].append(record)

        for csv_path in prime_csv:
            _group = prime_csv[csv_path]
            res = self.opt_matching(_group)
            self.form_index_file(res)

    def get_shape_inputs(self, path_list):
        _path_list = shape_from_extractor_args(path_list)
        return _path_list

    def opt_matching(self, group: List[dict]):
        # collect speakerID
        ids = [_k for _k in group[0].keys() if not _k in ("__name__", "__pair__")]

        prime_id = {}
        for sp_id in ids:
            prime_id[sp_id] = []

            for record in group:
                name = record["__name__"]
                pair = record["__pair__"]
                score = record[sp_id]["volatility"] / record[sp_id]["data_num"]
                used_rate = record[sp_id]["data_num"] / record[sp_id]["all_data"]
                prime_id[sp_id].append((name, pair, score, used_rate))

        # sort

        return 1

    def form_index_file(self, match_res):
        pass
