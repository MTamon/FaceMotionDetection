"""This program is for much audio-visual data"""

from logging import Logger
from typing import List
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
import re

from src.utils import batching
from src.io import load_shaped, load_luu_csv


class MatchAV:
    def __init__(self, logger: Logger, batch_size: int = 5):
        self.logger = logger
        self.batch_size = batch_size

    def __call__(
        self, match_datas: List[dict], shape_result_path: List[str], multi_proc=True
    ) -> List[dict]:
        seq_input = self.concatenate_inputs(match_datas, shape_result_path)
        seq_input = batching(seq_input, self.batch_size)

        all_process = len(match_datas)

        results = []

        for idx, batch in enumerate(seq_input):
            self.logger.info(f" >> Progress: {(idx+1)}/{all_process} << ")
            batch[0][2] = True  # tqdm ON

            if multi_proc:
                with Pool(processes=None) as pool:
                    results += pool.starmap(self.phase, batch)
            else:
                for _ba in batch:
                    results.append(self.phase(*_ba))

        self.logger.info(" >> DONE. << ")

        return results

    def phase(self, match_info: dict, shape_path: str, tqdm_visualize=False):
        # match_datas shape {".csv": path, ".wav": [path1, ...]}
        # shape_result shape [shape_result: ndarray, norm_info, normalizer, fps] -> [0] & [3]

        csv_path = match_info[".csv"]["path"]
        csv_path = "/".join(re.split(r"[\\]", csv_path))

        event_list = load_luu_csv(csv_path)
        all_shape_result = load_shaped(shape_path)

        shape_result = "/".join(re.split(r"[\\]", all_shape_result[0]))
        fps = all_shape_result[3]
        data_info = all_shape_result[4]

        measure_res = {
            "__name__": csv_path,
            "__pair__": shape_path,
            "__max__": data_info["max_length"],
            "__able__": data_info["available"],
        }

        if tqdm_visualize:
            iterator = tqdm(event_list, desc="  measure mouth ")
        else:
            iterator = event_list

        for event in iterator:
            start = int((event["startTime"] - 0.5) * fps)
            end = int((event["endTime"] + 0.5) * fps)
            sp_id = event["speakerID"].split("_")[0]  # part of ICXX

            if sp_id[:2] != "IC":
                continue

            if not sp_id in measure_res.keys():
                measure_res[sp_id] = {
                    "volatility": 0.0,
                    "data_num": 0,
                    "all_data": 0,
                }

            target = shape_result[start : end + 1]
            volatility, data_num = self.measure_mouth_movement(target)
            measure_res[sp_id]["volatility"] += volatility
            measure_res[sp_id]["data_num"] += data_num
            measure_res[sp_id]["all_data"] += end - start

        return measure_res

    def concatenate_inputs(self, sq1, sq2) -> list:
        if len(sq1) != len(sq2):
            l1, l2 = len(sq1), len(sq2)
            raise ValueError(
                f"Input sequences must be same length. But in this case length {l1} & {l2}."
            )

        _sq = []
        for s1, s2 in zip(sq1, sq2):
            _sq.append([s1, s2, False])

        return _sq

    def measure_mouth_movement(self, target):
        dt_len = 0
        volatility = 0
        prevs = None

        for _step in target:
            face = _step["countenance"]
            if face is None:
                prevs = None
                continue

            v_pts = np.concatenate((face[0].reshape(1, -1), face[11:18]))
            _v_pts = np.concatenate((face[11:18], face[0].reshape(1, -1)))
            pts_len = np.linalg.norm(v_pts - _v_pts, axis=-1)

            if prevs is None:
                prevs = pts_len
                continue

            volatility += np.sum(abs(pts_len - prevs))
            dt_len += 1

        return (volatility, dt_len)
