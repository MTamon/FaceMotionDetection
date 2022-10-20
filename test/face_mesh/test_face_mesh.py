"""Thit is test program"""

import _path
from logging import Logger

from logger_gen import set_logger
from src.face_mesh.face_mesh import HeadPoseEstimation
from src.io import load_face_area


def process(logger: Logger, paths):
    # prepare
    input_paths = []
    input_areas = []
    for input_v, hp_path, output, input_a in paths:
        input_areas.append(load_face_area(input_a))
        input_paths.append((input_v, hp_path, output))

    visualize = True
    hpe = HeadPoseEstimation(
        logger=logger,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_face=1,
        visualize=visualize,
        result_length=1000000,
    )

    # run test code
    hp_paths = hpe(input_paths, input_areas)

    for hpp in hp_paths:
        for hp in hpp:
            logger.info(f"saved file {hp}")


if __name__ == "__main__":
    logger = set_logger("TEST-HPE", "log/test/test-hpe.log")
    paths = [
        (
            "./data/test/webcame.mp4",
            "test/face_mesh/out/webcame.hp",
            "test/face_mesh/out/webcame.mp4",
            "test/trim/out/result/webcame.area",
        ),
        (
            "./data/test/webcame2.mp4",
            "test/face_mesh/out/webcame2.hp",
            "test/face_mesh/out/webcame2.mp4",
            "test/trim/out/result/webcame2.area",
        ),
        (
            "./data/test/short1.mp4",
            "test/face_mesh/out/short1.hp",
            "test/face_mesh/out/short1.mp4",
            "test/trim/out/result/short1.area",
        ),
        (
            "./data/test/short2.mp4",
            "test/face_mesh/out/short2.hp",
            "test/face_mesh/out/short2.mp4",
            "test/trim/out/result/short2.area",
        ),
        # (
        #     "./data/test/midol1s.mp4",
        #     "test/face_mesh/out/midol1s.hp",
        #     "test/face_mesh/out/midol1s.mp4",
        #     "test/trim/out/result/midol1s.area",
        # ),
        # (
        #     "./data/test/test1.mp4",
        #     "test/face_mesh/out/test1.hp",
        #     "test/face_mesh/out/test1.mp4",
        #     "test/trim/out/result/test1.area",
        # ),
        # (
        #     "./data/test/test2.mp4",
        #     "test/face_mesh/out/test2.hp",
        #     "test/face_mesh/out/test2.mp4",
        #     "test/trim/out/result/test2.area",
        # ),
        # (
        #     "./data/test/test3.mp4",
        #     "test/face_mesh/out/test3.hp",
        #     "test/face_mesh/out/test3.mp4",
        #     "test/trim/out/result/test3.area",
        # ),
        # (
        #     "./data/test/test4.mp4",
        #     "test/face_mesh/out/test4.hp",
        #     "test/face_mesh/out/test4.mp4",
        #     "test/trim/out/result/test4.area",
        # ),
    ]
    process(logger, paths)
