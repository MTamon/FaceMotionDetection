"""Thit is test program"""

from _path import SYSTEM_AREA
from logging import Logger

from logger_gen import set_logger
from src.face_mesh.face_mesh import HeadPoseEstimation
from src.io import load_face_area

if __name__ == "__main__":
    print(f"test_face_mesh: connect to {SYSTEM_AREA}")


def process(logger: Logger, paths):

    visualize = False
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
        #     "./data/test/webcame3.mp4",
        #     "test/face_mesh/out/webcame3.hp",
        #     load_face_area("test/trim/out/result/webcame3.area"),
        #     "test/face_mesh/out/webcame3.mp4",
        # ),
        # (
        #     "./data/test/webcame4.mp4",
        #     "test/face_mesh/out/webcame4.hp",
        #     load_face_area("test/trim/out/result/webcame4.area"),
        #     "test/face_mesh/out/webcame4.mp4",
        # ),
        # (
        #     "./data/test/webcame5.mp4",
        #     "test/face_mesh/out/webcame5.hp",
        #     load_face_area("test/trim/out/result/webcame5.area"),
        #     "test/face_mesh/out/webcame5.mp4",
        # ),
        # (
        #     "./data/test/webcame6.mp4",
        #     "test/face_mesh/out/webcame6.hp",
        #     load_face_area("test/trim/out/result/webcame6.area"),
        #     "test/face_mesh/out/webcame6.mp4",
        # ),
        # (
        #     "./data/test/webcame7.mp4",
        #     "test/face_mesh/out/webcame7.hp",
        #     load_face_area("test/trim/out/result/webcame7.area"),
        #     "test/face_mesh/out/webcame7.mp4",
        # ),
        # (
        #     "./data/test/webcame8.mp4",
        #     "test/face_mesh/out/webcame8.hp",
        #     load_face_area("test/trim/out/result/webcame8.area"),
        #     "test/face_mesh/out/webcame8.mp4",
        # ),
        # (
        #     "./data/test/webcame9.mp4",
        #     "test/face_mesh/out/webcame9.hp",
        #     load_face_area("test/trim/out/result/webcame9.area"),
        #     "test/face_mesh/out/webcame9.mp4",
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
        # (
        #     "./data/test/test2.mp4",
        #     "test/face_mesh/out/test2.hp",
        #     load_face_area("test/trim/out/result/test2.area"),
        #     "test/face_mesh/out/test2.mp4",
        # ),
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
        (
            "./data/test/test5.mp4",
            "test/face_mesh/out/test5.hp",
            load_face_area("test/trim/out/result/test5.area"),
            "test/face_mesh/out/test5.mp4",
        ),
        (
            "./data/test/test6.mp4",
            "test/face_mesh/out/test6.hp",
            load_face_area("test/trim/out/result/test6.area"),
            "test/face_mesh/out/test6.mp4",
        ),
        (
            "./data/test/test7.mp4",
            "test/face_mesh/out/test7.hp",
            load_face_area("test/trim/out/result/test7.area"),
            "test/face_mesh/out/test7.mp4",
        ),
        (
            "./data/test/test8.mp4",
            "test/face_mesh/out/test8.hp",
            load_face_area("test/trim/out/result/test8.area"),
            "test/face_mesh/out/test8.mp4",
        ),
        (
            "./data/test/test9.mp4",
            "test/face_mesh/out/test9.hp",
            load_face_area("test/trim/out/result/test9.area"),
            "test/face_mesh/out/test9.mp4",
        ),
        (
            "./data/test/test10.mp4",
            "test/face_mesh/out/test10.hp",
            load_face_area("test/trim/out/result/test10.area"),
            "test/face_mesh/out/test10.mp4",
        ),
        (
            "./data/test/test11.mp4",
            "test/face_mesh/out/test11.hp",
            load_face_area("test/trim/out/result/test11.area"),
            "test/face_mesh/out/test11.mp4",
        ),
        (
            "./data/test/test12.mp4",
            "test/face_mesh/out/test12.hp",
            load_face_area("test/trim/out/result/test12.area"),
            "test/face_mesh/out/test12.mp4",
        ),
        (
            "./data/test/test13.mp4",
            "test/face_mesh/out/test13.hp",
            load_face_area("test/trim/out/result/test13.area"),
            "test/face_mesh/out/test13.mp4",
        ),
        (
            "./data/test/test14.mp4",
            "test/face_mesh/out/test14.hp",
            load_face_area("test/trim/out/result/test14.area"),
            "test/face_mesh/out/test14.mp4",
        ),
    ]
    process(log, path_list)
