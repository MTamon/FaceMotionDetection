"""This code is for integrated post-processing"""
import os
import re

from logging import Logger
from argparse import Namespace
from typing import List, Tuple
from collections import OrderedDict
from pprint import pformat
from multiprocessing import Pool
from tqdm import tqdm

from .shape.shaper import Shaper
from .match.matching import MatchAV
from .utils import shape_from_extractor_args, batching
from .io import write_index_file, load_index_file, load_measure_mouth
from .visualizer import Visualizer


class CEJC_Builder:
    def __init__(self, logger: Logger, args: Namespace):
        self.logger = logger
        self.args = args

        shaper_args = self.make_args_shaper(logger, args)
        match_args = self.make_args_match(logger, args)

        self.shaper = Shaper(**shaper_args)
        self.merger = MatchAV(**match_args)

        self.batch_size = self.shaper.batch_size

        self.threshold_len = args.threshold_len
        self.threshold_use = args.threshold_use
        self.visualize_match = args.visualize_match
        self.single_proc = args.single_proc_matching

    def __call__(self, shaper_list: list, matchav_list: list):
        shaper_list = batching(shaper_list, self.batch_size)
        matchav_list = batching(matchav_list, self.batch_size)

        merge_res = []
        all_process = len(shaper_list)

        for i, (batch_s, batch_m) in enumerate(zip(shaper_list, matchav_list)):
            self.logger.info(f"BULDING {i}/{all_process}")

            shape_r = self.shaper(batch_s)

            # batch_m shape {".csv": path, ".wav": [path1, ...]} * batch_size
            # shape_r shape [shape_result: ndarray, norm_info, normalizer] * batch_size
            merge_res += self.merger(batch_m, shape_r)

            self.logger.info("BULDING Done.")

        _merge_res = []
        for path in merge_res:
            # Rejected data
            if path is not None:
                _merge_res.append(load_measure_mouth(path))
        merge_res = _merge_res

        self.logger.info("BUILDER >> Analysis process done.")
        self.logger.info("BUILDER >> Start Optimize process.")

        prime_csv = {}
        for record in merge_res:
            if not record["__name__"] in prime_csv.keys():
                prime_csv[record["__name__"]] = []

            prime_csv[record["__name__"]].append(record)

        phase_args = []
        for csv_path in prime_csv:
            phase_args.append([csv_path, prime_csv, False])
        phase_args = batching(phase_args, self.batch_size)

        results = []

        for batch in phase_args:
            if not self.single_proc:
                batch[0][2] = True
                with Pool(processes=None) as pool:
                    results += pool.starmap(self.phase, batch)
            else:
                for _ba in batch:
                    _ba[2] = True
                    results.append(self.phase(*_ba))

        # display result
        for save_path in results:
            match_info = load_index_file(save_path)

            fp = pformat(match_info).split("\n")
            print("\n".join(fp))
            for _fp in fp:
                self.logger.info(_fp)

    def phase(self, csv_path, prime_csv, tqdm_visual=False):
        _group = prime_csv[csv_path]
        match_info = self.opt_matching(_group, tqdm_visual)
        save_path = write_index_file(match_info)

        if self.visualize_match:
            Visualizer.audio_visual_matching(self.logger, match_info)

        return save_path

    def get_shape_inputs(self, path_list):
        _path_list = shape_from_extractor_args(path_list)
        return _path_list

    def opt_matching(
        self, group: List[dict], tqdm_visual=False
    ) -> List[Tuple[str, str]]:
        # collect speakerID
        info_keys = ("__name__", "__pair__", "__max__", "__able__")
        ids = [_k for _k in group[0].keys() if not _k in info_keys]

        if tqdm_visual:
            iterator = tqdm(ids, desc="   opt-matching ")
        else:
            iterator = ids

        prime_id = OrderedDict()
        for sp_id in iterator:
            prime_id[sp_id] = []

            for record in group:
                name = record["__name__"]
                pair = record["__pair__"]
                length_max = record["__max__"]
                volatility = record[sp_id]["volatility"]
                if record[sp_id]["data_num"] > record[sp_id]["fps"] * 10:
                    score = volatility / record[sp_id]["data_num"]
                    used_rate = record[sp_id]["data_num"] / record[sp_id]["all_data"]
                else:
                    score = 0
                    used_rate = 0
                if length_max > self.threshold_len and used_rate > self.threshold_use:
                    res_dict = {
                        "name": name,
                        "pair": pair,
                        "score": score,
                        "sp_id": sp_id,
                        "used": used_rate,
                    }
                    prime_id[sp_id].append(res_dict)
                else:
                    res_dict = {
                        "name": name,
                        "pair": pair,
                        "score": 0,
                        "sp_id": sp_id,
                        "used": used_rate,
                    }
                    prime_id[sp_id].append(res_dict)

        prime_id = self.softmax_par_pair(prime_id)
        prime_id = self.sort_prime_id(prime_id)

        # Search
        # index_list, _ = self.search_opt_comb(prime_id)
        index_list = [0] * len(prime_id)

        # form result
        res = []
        for idx, sp_id in zip(index_list, prime_id):
            name = prime_id[sp_id][idx]["name"]
            pair = prime_id[sp_id][idx]["pair"]
            scre = prime_id[sp_id][idx]["score"]
            _res = (sp_id, pair, name, scre)
            res.append(_res)

        match_info = self.form_index_file(res)

        return match_info

    def softmax_par_pair(self, prime_id):
        prime_pair = {}
        for _k in prime_id:
            for record in prime_id[_k]:
                pp_key = record["pair"]
                if pp_key in prime_pair:
                    prime_pair[pp_key].append(record)
                else:
                    prime_pair[pp_key] = [record]
        for _k in prime_pair:
            pair_sum = 1e-4
            for record in prime_pair[_k]:
                pair_sum += record["score"]
            for record in prime_pair[_k]:
                record["score"] /= pair_sum

        return prime_id

    def form_index_file(self, match_res):
        dir_name = os.path.basename(match_res[0][2]).split("-")[0]
        _index_file_name = dir_name + ".avidx"

        path = os.path.dirname(match_res[0][2])

        idx_path = os.path.join(path, _index_file_name)
        idx_path = "/".join(re.split(r"\\", idx_path))

        match_info = {"name": idx_path, "csv": match_res[0][2], "pairs": []}

        for (sp_id, pair, _, scr) in match_res:
            wav_path = os.path.join(path, "_".join([dir_name, sp_id + ".wav"]))
            wav_path = "/".join(re.split(r"\\", wav_path))
            if os.path.isfile(wav_path):
                match_info["pairs"].append((pair, wav_path, scr, sp_id))

        prime_pair = {}
        for info in match_info["pairs"]:
            if not info[0] in prime_pair:
                prime_pair[info[0]] = (info[1], info[2], info[3])
            else:
                if prime_pair[info[0]][1] < info[2]:
                    prime_pair[info[0]] = (info[1], info[2], info[3])
        match_info["pairs"] = [
            {"sh": pair, "wav": wav_path, "spID": sp_id}
            for pair, (wav_path, _, sp_id) in prime_pair.items()
        ]

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
            res.append((_index_list, scr))

            for i in range(len(_pattern_idx)):
                _pattern_idx[i] += 1
                if _pattern_idx[i] == len(_pattern[i]):
                    _pattern_idx[i] = 0
                else:
                    break

            first_flg = False

        return sorted(res, key=lambda x: x[1], reverse=True)[0]

    def make_args_shaper(self, logger: Logger, args: Namespace):
        shaper_args = {}
        shaper_args["logger"] = logger
        shaper_args["order"] = args.order
        shaper_args["noise_subtract"] = args.noise_subtract
        shaper_args["mask_subtract"] = args.mask_subtract
        shaper_args["batch_size"] = args.batch_size
        shaper_args["visualize_graph"] = args.visualize_graph
        shaper_args["visualize_noise"] = args.visualize_noise
        shaper_args["visualize_interpolation"] = args.visualize_interpolation
        shaper_args["visualize_all"] = args.visualize_all
        shaper_args["visualize_front"] = args.visualize_front
        shaper_args["redo"] = args.redo_shaper
        shaper_args["single_proc"] = args.single_proc_shaper

        return shaper_args

    def make_args_match(self, logger: Logger, args: Namespace):
        match_args = {}
        match_args["logger"] = logger
        match_args["batch_size"] = args.batch_size
        match_args["method"] = args.measure_method
        match_args["redo"] = args.redo_matching
        match_args["single_proc"] = args.single_proc_matching

        return match_args
