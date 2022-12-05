"""Thit is test program"""

from _path import SYSTEM_AREA
import os
from logging import Logger
from typing import List
from pprint import pprint, pformat

from logger_gen import set_logger
from src.io import load_face_area, write_face_area
from src.trim.triming_area import TrimFace

if __name__ == "__main__":
    print(f"test_triming_area: connect to {SYSTEM_AREA}")


def process(logger: Logger, paths: List[str]):
    # prepare
    i_s = 1
    visualize = True
    trimer = TrimFace(
        logger=logger,
        min_detection_confidence=0.6,
        model_selection=1,
        frame_step=1,
        box_ratio=1.1,
        track_volatility=0.3,
        lost_volatility=0.1,
        size_volatility=0.03,
        sub_track_volatility=1.0,
        sub_size_volatility=0.5,
        threshold=0.1,
        threshold_size_rate=2,
        overlap=0.8,
        integrate_step=i_s,
        integrate_volatility=0.4,
        use_tracking=True,
        prohibit_integrate=1.1,
        size_limit_rate=4,
        gc=0.03,
        gc_term=300,
        gc_success=0.1,
        lost_track=2,
        process_num=3,
        redo=True,
        visualize=visualize,
        single_process=True,
    )

    # run test code
    results = trimer(paths)

    pprint(results)
    for line in pformat(results).split("\n"):
        logger.info(line)

    # check pickle's save & load function
    for idx, (result, fpath) in enumerate(zip(results, paths)):
        compatible_face = write_face_area(paths[idx][2], result)
        _compatible_face = load_face_area(paths[idx][2])

        name = os.path.basename(fpath[0])

        success = 0
        for (face, _face) in zip(compatible_face, _compatible_face):
            if face == _face:
                success += 1
        logger.info(f"{name} / success save & load result : {success}/{len(result)}")


if __name__ == "__main__":
    log = set_logger("TEST-TRIM", "log/test/test-trim.log")
    paths_input = [
        # (
        #     "./data/test/webcame.mp4",
        #     "test/trim/out/result/webcame.mp4",
        #     "test/trim/out/result/webcame.area",
        # ),
        # (
        #     "./data/test/webcame2.mp4",
        #     "test/trim/out/result/webcame2.mp4",
        #     "test/trim/out/result/webcame2.area",
        # ),
        # (
        #     "./data/test/webcame3.mp4",
        #     "test/trim/out/result/webcame3.mp4",
        #     "test/trim/out/result/webcame3.area",
        # ),
        # (
        #     "./data/test/webcame4.mp4",
        #     "test/trim/out/result/webcame4.mp4",
        #     "test/trim/out/result/webcame4.area",
        # ),
        # (
        #     "./data/test/webcame5.mp4",
        #     "test/trim/out/result/webcame5.mp4",
        #     "test/trim/out/result/webcame5.area",
        # ),
        # (
        #     "./data/test/webcame6.mp4",
        #     "test/trim/out/result/webcame6.mp4",
        #     "test/trim/out/result/webcame6.area",
        # ),
        # (
        #     "./data/test/webcame7.mp4",
        #     "test/trim/out/result/webcame7.mp4",
        #     "test/trim/out/result/webcame7.area",
        # ),
        # (
        #     "./data/test/webcame8.mp4",
        #     "test/trim/out/result/webcame8.mp4",
        #     "test/trim/out/result/webcame8.area",
        # ),
        # (
        #     "./data/test/webcame9.mp4",
        #     "test/trim/out/result/webcame9.mp4",
        #     "test/trim/out/result/webcame9.area",
        # ),
        # (
        #     "./data/test/short1.mp4",
        #     "test/trim/out/result/short1.mp4",
        #     "test/trim/out/result/short1.area",
        # ),
        # (
        #     "./data/test/short2.mp4",
        #     "test/trim/out/result/short2.mp4",
        #     "test/trim/out/result/short2.area",
        # ),
        # (
        #     "./data/test/midol1s.mp4",
        #     "test/trim/out/result/midol1s.mp4",
        #     "test/trim/out/result/midol1s.area",
        # ),
        (
            "./data/test/test1.mp4",
            "test/trim/out/result/test1.mp4",
            "test/trim/out/result/test1.area",
        ),
        # (
        #     "./data/test/test2.mp4",
        #     "test/trim/out/result/test2.mp4",
        #     "test/trim/out/result/test2.area",
        # ),
        # (
        #     "./data/test/test3.mp4",
        #     "test/trim/out/result/test3.mp4",
        #     "test/trim/out/result/test3.area",
        # ),
        # (
        #     "./data/test/test4.mp4",
        #     "test/trim/out/result/test4.mp4",
        #     "test/trim/out/result/test4.area",
        # ),
        # (
        #     "./data/test/test5.mp4",
        #     "test/trim/out/result/test5.mp4",
        #     "test/trim/out/result/test5.area",
        # ),
        # (
        #     "./data/test/test6.mp4",
        #     "test/trim/out/result/test6.mp4",
        #     "test/trim/out/result/test6.area",
        # ),
        # (
        #     "./data/test/test7.mp4",
        #     "test/trim/out/result/test7.mp4",
        #     "test/trim/out/result/test7.area",
        # ),
        # (
        #     "./data/test/test8.mp4",
        #     "test/trim/out/result/test8.mp4",
        #     "test/trim/out/result/test8.area",
        # ),
        # (
        #     "./data/test/test9.mp4",
        #     "test/trim/out/result/test9.mp4",
        #     "test/trim/out/result/test9.area",
        # ),
        # (
        #     "./data/test/test10.mp4",
        #     "test/trim/out/result/test10.mp4",
        #     "test/trim/out/result/test10.area",
        # ),
        # (
        #     "./data/test/test11.mp4",
        #     "test/trim/out/result/test11.mp4",
        #     "test/trim/out/result/test11.area",
        # ),
        # (
        #     "./data/test/test12.mp4",
        #     "test/trim/out/result/test12.mp4",
        #     "test/trim/out/result/test12.area",
        # ),
        # (
        #     "./data/test/test13.mp4",
        #     "test/trim/out/result/test13.mp4",
        #     "test/trim/out/result/test13.area",
        # ),
        # (
        #     "./data/test/test14.mp4",
        #     "test/trim/out/result/test14.mp4",
        #     "test/trim/out/result/test14.area",
        # ),
        # (
        #     "./data/test/test15.mp4",
        #     "test/trim/out/result/test15.mp4",
        #     "test/trim/out/result/test15.area",
        # ),
        # (
        #     "./data/test/test16.mp4",
        #     "test/trim/out/result/test16.mp4",
        #     "test/trim/out/result/test16.area",
        # ),
        # (
        #     "./data/test/test17.mp4",
        #     "test/trim/out/result/test17.mp4",
        #     "test/trim/out/result/test17.area",
        # ),
    ]
    process(log, paths_input)
