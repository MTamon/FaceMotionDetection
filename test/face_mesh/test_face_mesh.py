"""Thit is test program"""

from logging import Logger

from logger_gen import set_logger
from src.face_mesh.face_mesh import HeadPoseEstimation
from src.io import load_face_area

import _path


def process(logger: Logger, paths):
    for input_v, input_a, output, hp_path in paths:
        # prepare
        areas = load_face_area(input_a)
        visualize = True
        hpe = HeadPoseEstimation(
            logger=logger,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_face=1,
            visualize=visualize,
            result_path=hp_path,
            result_length=10000,
        )

        # run test code
        hp_paths = hpe(input_v, areas, output)

        for hpp in hp_paths:
            logger.info(f"saved file {hpp}")

        logger.info("############################################")


if __name__ == "__main__":
    logger = set_logger("TEST-HPE", "log/test/test-hpe.log")
    paths = [
        (
            "./data/test/webcame.mp4",
            "test/trim/out/result/webcame.area",
            "test/face_mesh/out/webcame.mp4",
            "test/face_mesh/out/webcame.hp",
        ),
        (
            "./data/test/midol1s.mp4",
            "test/trim/out/result/midol1s.area",
            "test/face_mesh/out/midol1s.mp4",
            "test/face_mesh/out/midol1s.hp",
        ),
        (
            "./data/test/short1.mp4",
            "test/trim/out/result/short1.area",
            "test/face_mesh/out/short1.mp4",
            "test/face_mesh/out/short1.hp",
        ),
        (
            "./data/test/short2.mp4",
            "test/trim/out/result/short2.area",
            "test/face_mesh/out/short2.mp4",
            "test/face_mesh/out/short2.hp",
        ),
        (
            "./data/test/test1.mp4",
            "test/trim/out/result/test1.area",
            "test/face_mesh/out/test1.mp4",
            "test/face_mesh/out/test1.hp",
        ),
        (
            "./data/test/test2.mp4",
            "test/trim/out/result/test2.area",
            "test/face_mesh/out/test2.mp4",
            "test/face_mesh/out/test2.hp",
        ),
        (
            "./data/test/test3.mp4",
            "test/trim/out/result/test3.area",
            "test/face_mesh/out/test3.mp4",
            "test/face_mesh/out/test3.hp",
        ),
        (
            "./data/test/test4.mp4",
            "test/trim/out/result/test4.area",
            "test/face_mesh/out/test4.mp4",
            "test/face_mesh/out/test4.hp",
        ),
    ]
    process(logger, paths)
