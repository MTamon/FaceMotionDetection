import pickle
from typing import List

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
HWAD_POSE_KEYS = ["step", "area", "origin", "angles", "landmarks", "activation"]


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
    head_pose = None
    with open(path, "rb") as f:
        head_pose = pickle.load(f)

    return head_pose


class InvalidDictError(Exception):
    pass
