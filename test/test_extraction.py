"""Thit is test program"""

from argparse import Namespace
import _path
import os
from logging import Logger
from typing import List

from logger_gen import set_logger
from src.io import load_face_area, write_face_area
from src.trim.triming_area import TrimFace


def process(logger: Logger, args: Namespace, paths: List[str]):
    # prepare
    pass


if __name__ == "__main__":
    log = set_logger("TEST-EXTRACT", "log/test/test-extract.log")
    paths_input = [
        (
            "./data/test/webcame.mp4",
            "test/out/webcame.hp",
            "test/out/webcame.area",
            "test/out/webcame.mp4",
        ),
        (
            "./data/test/webcame2.mp4",
            "test/out/webcame2.hp",
            "test/out/webcame2.area",
            "test/out/webcame2.mp4",
        ),
        (
            "./data/test/short1.mp4",
            "test/out/short1.hp",
            "test/out/short1.area",
            "test/out/short1.mp4",
        ),
        (
            "./data/test/short2.mp4",
            "test/out/short2.hp",
            "test/out/short2.area",
            "test/out/short2.mp4",
        ),
        # (
        #     "./data/test/midol1s.mp4",
        #     "test/out/midol1s.hp",
        #     "test/out/midol1s.area",
        #     "test/out/midol1s.mp4",
        # ),
        # (
        #     "./data/test/test1.mp4",
        #     "test/out/test1.hp",
        #     "test/out/test1.area",
        #     "test/out/test1.mp4",
        # ),
        # (
        #     "./data/test/test2.mp4",
        #     "test/out/test2.hp",
        #     "test/out/test2.area",
        #     "test/out/test2.mp4",
        # ),
        # (
        #     "./data/test/test3.mp4",
        #     "test/out/test3.hp",
        #     "test/out/test3.area",
        #     "test/out/test3.mp4",
        # ),
        # (
        #     "./data/test/test4.mp4",
        #     "test/out/test4.hp",
        #     "test/out/test4.area",
        #     "test/out/test4.mp4",
        # ),
    ]
    process(log, paths_input)
