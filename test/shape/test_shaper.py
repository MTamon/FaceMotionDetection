"""Thit is test program"""

from typing import List
from _path import SYSTEM_AREA
from logging import Logger

from logger_gen import set_logger
from src.shape.shaper import Shaper

import os

if __name__ == "__main__":
    print(f"test_shaper: connect to {SYSTEM_AREA}")


def process(logger: Logger, paths):

    shaping = Shaper(
        logger=logger,
        batch_size=3,
        visualize_graph=False,
        visualize_noise=False,
        visualize_interpolation=False,
        visualize_all=False,
        single_proc=False,
        redo=True,
    )

    paths = path_creater(paths)

    # run test code
    shaping(paths)


def path_creater(paths):
    new_paths = []
    for path in paths:
        ret, same_list = get_same_files(path[0])

        if not ret:
            continue

        for same_file in same_list:

            f_name = os.path.basename(same_file)
            out_dir = os.path.dirname(path[2])
            out_file = ".".join([f_name.split(".")[0], "sh"])
            out_path = os.path.join(out_dir, out_file)

            new_paths.append(
                (
                    same_file,
                    path[1],
                    out_path,
                )
            )
    return new_paths


def get_same_files(path) -> List[str]:
    """Check if the results of trim-area and face-mesh exist."""
    dir_path = os.path.dirname(path)
    target_file = os.path.basename(path)

    target_name, target_ext = target_file.split(".")
    target_name = target_name.split("_")

    same_list = []

    file_list = os.listdir(dir_path)
    exist_flg = False
    for file in file_list:
        if not os.path.isfile(os.path.join(dir_path, file)):
            continue

        f_name, f_ext = file.split(".")
        f_name = f_name.split("_")

        if f_ext != target_ext:
            continue

        flg = False
        for idx, t_n in enumerate(target_name):
            if t_n != f_name[idx]:
                flg = True
                break
        if not flg:
            exist_flg = True
            same_list.append(os.path.join(dir_path, file))

    return exist_flg, same_list


if __name__ == "__main__":
    log = set_logger("TEST-SHAPE", "log/test/test-shp.log")
    path_list = [
        (
            "test/face_mesh/out/webcame.hp",
            "./data/test/webcame.mp4",
            "test/shape/out/webcame.sh",
        ),
        (
            "test/face_mesh/out/webcame2.hp",
            "./data/test/webcame2.mp4",
            "test/shape/out/webcame2.sh",
        ),
        (
            "test/face_mesh/out/webcame3.hp",
            "./data/test/webcame3.mp4",
            "test/shape/out/webcame3.sh",
        ),
        (
            "test/face_mesh/out/webcame4.hp",
            "./data/test/webcame4.mp4",
            "test/shape/out/webcame4.sh",
        ),
        (
            "test/face_mesh/out/webcame5.hp",
            "./data/test/webcame5.mp4",
            "test/shape/out/webcame5.sh",
        ),
        (
            "test/face_mesh/out/webcame6.hp",
            "./data/test/webcame6.mp4",
            "test/shape/out/webcame6.sh",
        ),
        # (
        #     "test/face_mesh/out/webcame7.hp",
        #     "./data/test/webcame7.mp4",
        #     "test/shape/out/webcame7.sh",
        # ),
        # (
        #     "test/face_mesh/out/webcame8.hp",
        #     "./data/test/webcame8.mp4",
        #     "test/shape/out/webcame8.sh",
        # ),
        # (
        #     "test/face_mesh/out/webcame9.hp",
        #     "./data/test/webcame9.mp4",
        #     "test/shape/out/webcame9.sh",
        # ),
        # (
        #     "test/face_mesh/out/short1.hp",
        #     "./data/test/short1.mp4",
        #     "test/shape/out/short1.sh",
        # ),
        # (
        #     "test/face_mesh/out/short2.hp",
        #     "./data/test/short2.mp4",
        #     "test/shape/out/short2.sh",
        # ),
        # (
        #     "test/face_mesh/out/midol1s.hp",
        #     "./data/test/midol1s.mp4",
        #     "test/shape/out/midol1s.sh",
        # ),
        # (
        #     "test/face_mesh/out/test1.hp",
        #     "./data/test/test1.mp4",
        #     "test/shape/out/test1.sh",
        # ),
        # (
        #     "test/face_mesh/out/test2.hp",
        #     "./data/test/test2.mp4",
        #     "test/shape/out/test2.sh",
        # ),
        # (
        #     "test/face_mesh/out/test3.hp",
        #     "./data/test/test3.mp4",
        #     "test/shape/out/test3.sh",
        # ),
        # (
        #     "test/face_mesh/out/test4.hp",
        #     "./data/test/test4.mp4",
        #     "test/shape/out/test4.sh",
        # ),
        # (
        #     "test/face_mesh/out/test5.hp",
        #     "./data/test/test5.mp4",
        #     "test/shape/out/test5.sh",
        # ),
        # (
        #     "test/face_mesh/out/test6.hp",
        #     "./data/test/test6.mp4",
        #     "test/shape/out/test6.sh",
        # ),
        # (
        #     "test/face_mesh/out/test7.hp",
        #     "./data/test/test7.mp4",
        #     "test/shape/out/test7.sh",
        # ),
        # (
        #     "test/face_mesh/out/test8.hp",
        #     "./data/test/test8.mp4",
        #     "test/shape/out/test8.sh",
        # ),
        # (
        #     "test/face_mesh/out/test9.hp",
        #     "./data/test/test9.mp4",
        #     "test/shape/out/test9.sh",
        # ),
        # (
        #     "test/face_mesh/out/test10.hp",
        #     "./data/test/test10.mp4",
        #     "test/shape/out/test10.sh",
        # ),
        # (
        #     "test/face_mesh/out/test11.hp",
        #     "./data/test/test11.mp4",
        #     "test/shape/out/test11.sh",
        # ),
        # (
        #     "test/face_mesh/out/test12.hp",
        #     "./data/test/test12.mp4",
        #     "test/shape/out/test12.sh",
        # ),
        # (
        #     "test/face_mesh/out/test13.hp",
        #     "./data/test/test13.mp4",
        #     "test/shape/out/test13.sh",
        # ),
        # (
        #     "test/face_mesh/out/test14.hp",
        #     "./data/test/test14.mp4",
        #     "test/shape/out/test14.sh",
        # ),
    ]
    process(log, path_list)
