"""Thit is test program"""

from _path import SYSTEM_AREA
from logging import Logger

from logger_gen import set_logger
from src.shape.shaper import Shaper

print(f"test_shaper: connect to {SYSTEM_AREA}")


def process(logger: Logger, paths):
    # prepare
    # input_paths = []
    # input_areas = []
    # for input_v, hp_path, output, input_a in paths:
    #     input_areas.append(load_face_area(input_a))
    #     input_paths.append((input_v, hp_path, output))

    visualize = True
    shaping = Shaper(
        logger=logger,
        batch_size=3,
        visualize=visualize,
    )

    # run test code
    shaping(paths)


if __name__ == "__main__":
    log = set_logger("TEST-HPE", "log/test/test-hpe.log")
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
    ]
    process(log, path_list)
