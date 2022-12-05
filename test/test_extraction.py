"""Thit is test program"""

from argparse import Namespace
from logging import Logger
from typing import List

from _path import SYSTEM_AREA
from logger_gen import set_logger
from argments import get_fm_args
from src.extraction import Extraction

if __name__ == "__main__":
    print(f"test_extraction: connect to {SYSTEM_AREA}")


def process(logger: Logger, args: Namespace, paths: List[str]):
    # prepare
    extractor = Extraction(logger, args)

    extractor(paths)


if __name__ == "__main__":
    log = set_logger("TEST-EXTRACT", "log/test/test-extract.log")
    arguments = get_fm_args()
    paths_input = [
        (
            "./data/test/webcame.mp4",
            "test/out/webcame.hp",
            "test/out/webcame.area",
            "test/out/webcameT.mp4",
            "test/out/webcameF.mp4",
        ),
        (
            "./data/test/webcame2.mp4",
            "test/out/webcame2.hp",
            "test/out/webcame2.area",
            "test/out/webcame2T.mp4",
            "test/out/webcame2F.mp4",
        ),
        (
            "./data/test/webcame3.mp4",
            "test/out/webcame3.hp",
            "test/out/webcame3.area",
            "test/out/webcame3T.mp4",
            "test/out/webcame3F.mp4",
        ),
        # (
        #     "./data/test/short1.mp4",
        #     "test/out/short1.hp",
        #     "test/out/short1.area",
        #     "test/out/short1T.mp4",
        #     "test/out/short1F.mp4",
        # ),
        # (
        #     "./data/test/short2.mp4",
        #     "test/out/short2.hp",
        #     "test/out/short2.area",
        #     "test/out/short2T.mp4",
        #     "test/out/short2F.mp4",
        # ),
        # (
        #     "./data/test/midol1s.mp4",
        #     "test/out/midol1s.hp",
        #     "test/out/midol1s.area",
        #     "test/out/midol1sT.mp4",
        #     "test/out/midol1sF.mp4",
        # ),
        # (
        #     "./data/test/test1.mp4",
        #     "test/out/test1.hp",
        #     "test/out/test1.area",
        #     "test/out/test1T.mp4",
        #     "test/out/test1F.mp4",
        # ),
        # (
        #     "./data/test/test2.mp4",
        #     "test/out/test2.hp",
        #     "test/out/test2.area",
        #     "test/out/test2T.mp4",
        #     "test/out/test2F.mp4",
        # ),
        # (
        #     "./data/test/test3.mp4",
        #     "test/out/test3.hp",
        #     "test/out/test3.area",
        #     "test/out/test3T.mp4",
        #     "test/out/test3F.mp4",
        # ),
        # (
        #     "./data/test/test4.mp4",
        #     "test/out/test4.hp",
        #     "test/out/test4.area",
        #     "test/out/test4T.mp4",
        #     "test/out/test4F.mp4",
        # ),
        # (
        #     "./data/test/test18.mp4",
        #     "test/out/test18.hp",
        #     "test/out/test18.area",
        #     "test/out/test4T.mp4",
        #     "test/out/test4F.mp4",
        # ),
    ]
    process(log, arguments, paths_input)
