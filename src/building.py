"""This code is for integrated post-processing"""
import os
import re

from logging import Logger
from argparse import Namespace
from typing import List, Tuple
from collections import OrderedDict

from .shape.shaper import Shaper
from .match.matching import MatchAV
from .utils import shape_from_extractor_args, batching
from .io import write_index_file


class CEJC_Builder:
    def __init__(self, logger: Logger, args: Namespace):
        self.logger = logger
        self.args = args

        self.shaper = Shaper(logger)
        self.merger = MatchAV(logger)

        self.batch_size = self.shaper.batch_size

        self.threshold_len = args.threshold_len
        self.threshold_use = args.threshold_use

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
            match_info = self.opt_matching(_group)
            write_index_file(match_info)

    def get_shape_inputs(self, path_list):
        _path_list = shape_from_extractor_args(path_list)
        return _path_list

    def opt_matching(self, group: List[dict]) -> List[Tuple[str, str]]:
        # collect speakerID
        info_keys = ("__name__", "__pair__", "__max__", "__able__")
        ids = [_k for _k in group[0].keys() if not _k in info_keys]

        prime_id = OrderedDict()
        for sp_id in ids:
            prime_id[sp_id] = []

            for record in group:
                name = record["__name__"]
                pair = record["__pair__"]
                length_max = record["__max__"]
                volatility = record[sp_id]["volatility"]
                score = volatility / record[sp_id]["data_num"]
                used_rate = record[sp_id]["data_num"] / record[sp_id]["all_data"]
                if length_max > self.threshold_len and used_rate > self.threshold_use:
                    res_dict = {
                        "name": name,
                        "pair": pair,
                        "score": score,
                    }
                    prime_id[sp_id].append(res_dict)
                else:
                    res_dict = {
                        "name": name,
                        "pair": pair,
                        "score": 0,
                    }
                    prime_id[sp_id].append(res_dict)

        prime_id = self.sort_prime_id(prime_id)

        # Search
        index_list, _ = self.search_opt_comb(prime_id)

        # form result
        res = []
        for idx, sp_id in zip(index_list, prime_id):
            name = prime_id[sp_id][idx]["__name__"]
            pair = prime_id[sp_id][idx]["__pair__"]
            _res = (sp_id, pair, name)
            res.append(_res)

        match_info = self.form_index_file(res)

        return match_info

    def form_index_file(self, match_res):
        dir_name = os.path.basename(match_res[0][2]).split("-")
        _index_file_name = dir_name[0] + ".avidx"

        path = os.path.dirname(match_res[0][2])

        idx_path = os.path.join(path, "_".join([dir_name, _index_file_name]))
        idx_path = "/".join(re.split(r"\\", idx_path))

        match_info = {"name": idx_path, "pairs": []}

        for (sp_id, pair, _) in match_res:
            wav_path = os.path.join(path, "_".join([dir_name, sp_id + ".wav"]))
            wav_path = "/".join(re.split(r"\\", wav_path))
            match_info["pairs"].append((pair, wav_path))

        return match_info

    def sort_prime_id(self, prime_id: OrderedDict) -> OrderedDict:
        for sp_id in prime_id:
            prime_id[sp_id] = sorted(
                prime_id[sp_id], key=lambda x: x.get("score", 0), reverse=True
            )
        return prime_id

    def search_opt_comb(
        self, prime_id: OrderedDict, index_list=None
    ) -> Tuple[List[int], float]:
        if index_list is None:
            index_list = [0] * len(prime_id)
        assert len(index_list) == len(prime_id)

        duplication = {}

        # detect duplication
        for i, (index, sp_id) in enumerate(zip(index_list, prime_id)):
            pair = prime_id[sp_id][index]["pair"]

            if not pair in duplication:
                duplication[pair] = [i]
            else:
                duplication[pair].append(i)

        _pattern = []
        for pair in duplication:
            if len(duplication[pair]) == 1:
                continue

            _pattern.append(duplication[pair])

        # case: no-duplication
        if _pattern == []:
            scr = 0.0
            for sp_id, idx in zip(prime_id, index_list):
                scr += prime_id[sp_id][idx]["score"]
            return (index_list.copy(), scr)

        res = []
        _pattern_idx = [0] * len(_pattern)
        first_flg = True
        while sum(_pattern_idx) != 0 or first_flg:
            _index_list = index_list.copy()
            for i, idx in enumerate(_pattern_idx):
                _index_list[_pattern[i][idx]] += 1

            _index_list, scr = self.search_opt_comb(prime_id, _index_list)
            res.append((scr, _index_list))

            for i in range(len(_pattern_idx)):
                _pattern_idx[i] += 1
                if _pattern_idx[i] == len(_pattern[i]):
                    _pattern_idx[i] = 0
                else:
                    break

            first_flg = False

        return sorted(res, key=lambda x: x[1], reverse=True)[0]
