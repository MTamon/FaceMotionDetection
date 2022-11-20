"""This is test code"""

from _path import SYSTEM_AREA

from src.utils import Video
from logger_gen import set_logger

from tqdm import tqdm

if __name__ == "__main__":
    print(f"test_utils: connect to {SYSTEM_AREA}")

if __name__ == "__main__":
    # prepare
    log = set_logger("TEST-UTILS", "log/test/test-util.log")
    video = Video("./data/test/short1.mp4", "mp4v")

    for i, frame in enumerate(tqdm(video)):
        if frame is None:
            log.info("Over Iteration! %s/%s", str(i), str(video.cap_frames))
        else:
            continue
