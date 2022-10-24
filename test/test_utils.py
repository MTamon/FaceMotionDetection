import _path

from src.utils import Video
from logger_gen import set_logger

from tqdm import tqdm

if __name__ == "__main__":
    # prepare
    logger = set_logger("TEST-UTILS", "log/test/test-util.log")
    video = Video('./data/test/short1.mp4', 'mp4v')
    
    for i, frame in enumerate(tqdm(video)):
        if frame is None:
            logger.info('Over Iteration! '+str(i)+'/'+str(video.cap_frames))
        else:
            continue