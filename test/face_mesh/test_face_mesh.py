"""Thit is test program"""

from _path import SYSTEM_AREA
from logging import Logger

from logger_gen import set_logger
from src.face_mesh.face_mesh import HeadPoseEstimation
from src.io import load_face_area

print(f"test_face_mesh: connect to {SYSTEM_AREA}")


def process(logger: Logger, paths):
    # prepare
    # input_paths = []
    # input_areas = []
    # for input_v, hp_path, output, input_a in paths:
    #     input_areas.append(load_face_area(input_a))
    #     input_paths.append((input_v, hp_path, output))

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
    hp_paths = hpe(paths)

    for hpp in hp_paths:
        for hp in hpp:
            logger.info(f"saved file {hp}")


if __name__ == "__main__":
    log = set_logger("TEST-HPE", "log/test/test-hpe.log")
    path_list = [
        # (
        #     "./data/test/webcame.mp4",
        #     "test/face_mesh/out/webcame.hp",
        #     load_face_area("test/trim/out/result/webcame.area"),
        #     "test/face_mesh/out/webcame.mp4",
        # ),
        # (
        #     "./data/test/webcame2.mp4",
        #     "test/face_mesh/out/webcame2.hp",
        #     load_face_area("test/trim/out/result/webcame2.area"),
        #     "test/face_mesh/out/webcame2.mp4",
        # ),
        # (
        #     "./data/test/short1.mp4",
        #     "test/face_mesh/out/short1.hp",
        #     load_face_area("test/trim/out/result/short1.area"),
        #     "test/face_mesh/out/short1.mp4",
        # ),
        # (
        #     "./data/test/short2.mp4",
        #     "test/face_mesh/out/short2.hp",
        #     load_face_area("test/trim/out/result/short2.area"),
        #     "test/face_mesh/out/short2.mp4",
        # ),
        # (
        #     "./data/test/midol1s.mp4",
        #     "test/face_mesh/out/midol1s.hp",
        #     load_face_area("test/trim/out/result/midol1s.area"),
        #     "test/face_mesh/out/midol1s.mp4",
        # ),
        # (
        #     "./data/test/test1.mp4",
        #     "test/face_mesh/out/test1.hp",
        #     load_face_area("test/trim/out/result/test1.area"),
        #     "test/face_mesh/out/test1.mp4",
        # ),
        (
            "./data/test/test2.mp4",
            "test/face_mesh/out/test2.hp",
            load_face_area("test/trim/out/result/test2.area"),
            "test/face_mesh/out/test2.mp4",
        ),
        # (
        #     "./data/test/test3.mp4",
        #     "test/face_mesh/out/test3.hp",
        #     load_face_area("test/trim/out/result/test3.area"),
        #     "test/face_mesh/out/test3.mp4",
        # ),
        # (
        #     "./data/test/test4.mp4",
        #     "test/face_mesh/out/test4.hp",
        #     load_face_area("test/trim/out/result/test4.area"),
        #     "test/face_mesh/out/test4.mp4",
        # ),
    ]
    process(log, path_list)
