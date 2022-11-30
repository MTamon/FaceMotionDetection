import pickle
from typing import List
import re

import numpy as np
from numpy import ndarray

AREA_KEYS = [
    "birthtime",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "width",
    "height",
    "width_min",
    "height_min",
    "width_max",
    "height_max",
    "success",
]
HEAD_POSE_KEYS = [
    "step",
    "area",
    "resolution",
    "origin",
    "angles",
    "landmarks",
    "activation",
]

SHAPE_KEYS = [
    "step",
    "countenance",
    "rotate",
    "centroid",
    "ratio",
    "fsize",
    "noise",
    "masked",
    "ignore",
]


def write_face_area(path, face_area: List[dict]) -> List[dict]:
    # check dict keys
    new_face_area = []
    for area in face_area:
        new_area = {}
        for key in AREA_KEYS:
            if key not in area.keys():
                raise InvalidDictError("dictionary lucks key.")
            new_area[key] = area[key]
        new_face_area.append(new_area)

    # output by pickle
    with open(path, "wb") as f:
        pickle.dump(new_face_area, f)

    return new_face_area


def load_face_area(path) -> List[dict]:
    with open(path, "rb") as f:
        face_area = pickle.load(f)
    return face_area


def write_head_pose(path, head_pose: ndarray):
    # output by pickle
    with open(path, "wb") as f:
        pickle.dump(head_pose, f)


def load_head_pose(path) -> ndarray:
    with open(path, "rb") as f:
        head_pose = pickle.load(f)

    return head_pose


def write_shaped(path, shape_result: ndarray, norm_info, normalizer, fps) -> ndarray:
    # extruct key
    _shape_result = []
    for step_result in shape_result:
        _step_result = {}
        for key in SHAPE_KEYS:
            _step_result[key] = step_result[key]
        _shape_result.append(_step_result)
    shape_result = np.array(_shape_result)

    # output by pickle
    with open(path, "wb") as f:
        pickle.dump([shape_result, norm_info, normalizer, fps], f)

    return shape_result


def load_shaped(path) -> ndarray:
    with open(path, "rb") as f:
        shaped = pickle.load(f)
    return shaped


class InvalidDictError(Exception):
    pass


def load_luu_csv(path) -> List[dict]:
    def remove_tail_blank(split_csv: list):
        if split_csv[-1] == "":
            return split_csv[:-1]
        return split_csv

    def shaping(_record: str):
        _record = remove_tail_blank(_record.split(","))
        assert len(head_info) < len(_record)

        if len(head_info) > len(_record):
            dif_len = len(head_info) - len(_record)
            _record += [""] * dif_len

        return _record

    def convert_data(data: str):
        res = re.findall(r"([\d]+\.[\d]*|[\d]*\.[\d]+)", data)
        if res is not None and len(res) == 1:
            if res[0] == data:
                return float(data)
        res = re.findall(r"[\d]+", data)
        if res is not None and len(res) == 1:
            if res[0] == data:
                return int(data)
        return data

    res_dicts = []

    with open(path, "r", encoding="utf-8") as f:
        head_info = []
        for line_no, record in enumerate(f):
            if line_no == 0:
                head_info = remove_tail_blank(record.split(","))
                continue
            contents = shaping(record)

            res = {}
            for c, h in zip(contents, head_info):
                res[h] = convert_data(c)
            res_dicts.append(res)

    return res_dicts
