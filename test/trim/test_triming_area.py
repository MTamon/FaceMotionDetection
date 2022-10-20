"""Thit is test program"""

import os
from logging import Logger
from typing import List

from logger_gen import set_logger
from src.io import load_face_area, write_face_area
from src.trim.triming_area import TrimFace

import _path


def process(logger: Logger, paths: List[str]):
    # prepare
    i_s = 1
    visualize = True
    trimer = TrimFace(
        logger=logger,
        min_detection_confidence=0.7,
        model_selection=1,
        frame_step=1,
        box_ratio=1.1,
        track_volatility=0.3,
        lost_volatility=0.1,
        size_volatility=0.03,
        sub_track_volatility=1.0,
        sub_size_volatility=0.5,
        threshold=0.3,
        overlap=0.9,
        integrate_step=i_s,
        integrate_volatility=0.4,
        use_tracking=True,
        prohibit_integrate=0.7,
        size_limit_rate=4,
        gc=0.03,
        gc_term=100,
        gc_success=0.1,
        lost_track=2,
        process_num=3,
        visualize=visualize,
    )

    # run test code
    results = trimer(paths)

    # check pickle's save & load function
    for idx, (result, fpath) in enumerate(zip(results, paths)):
        compatible_face = write_face_area(paths[idx][2], result)
        _compatible_face = load_face_area(paths[idx][2])

        name = os.path.basename(fpath[0])

        success = 0
        for i, (face, _face) in enumerate(zip(compatible_face, _compatible_face)):
            if face == _face:
                success += 1
        logger.info(f"{name} / success save & load result : {success}/{len(result)}")


if __name__ == "__main__":
    log = set_logger("TEST-TRIM", "log/test/test-trim.log")
    paths_input = [
        (
            "./data/test/webcame.mp4",
            "test/trim/out/webcame.mp4",
            "test/trim/out/webcame.area",
        ),
        (
            "./data/test/short1.mp4",
            "test/trim/out/short1.mp4",
            "test/trim/out/short1.area",
        ),
        (
            "./data/test/short2.mp4",
            "test/trim/out/short2.mp4",
            "test/trim/out/short2.area",
        ),
        # (
        #     "./data/test/midol1s.mp4",
        #     "test/trim/out/midol1s.mp4",
        #     "test/trim/out/midol1s.area",
        # ),
        # (
        #     "./data/test/test1.mp4",
        #     "test/trim/out/test1.mp4",
        #     "test/trim/out/test1.area",
        # ),
        # (
        #     "./data/test/test2.mp4",
        #     "test/trim/out/test2.mp4",
        #     "test/trim/out/test2.area",
        # ),
        # (
        #     "./data/test/test3.mp4",
        #     "test/trim/out/test3.mp4",
        #     "test/trim/out/test3.area",
        # ),
        # (
        #     "./data/test/test4.mp4",
        #     "test/trim/out/test4.mp4",
        #     "test/trim/out/test4.area",
        # ),
    ]
    process(log, paths_input)
