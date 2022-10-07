import math

from pkg_resources import compatible_platforms
import _path

from src.trim.triming_area import TrimFace
from src.utils import Video
from src.visualizer import Visualizer
from src.io import write_face_area, load_face_area
from logger_gen import set_logger

from tqdm import tqdm
import cv2
import pickle

def process(logger, paths):
    for input, output, area_path in paths:
        # prepare
        video = Video(input, 'mp4v')
        i_t = int(len(video)/1000)
        i_s = math.ceil(len(video)/10000)
        trimer = TrimFace(min_detection_confidence=0.5, model_selection=1, logger=logger, \
            frame_step=1, box_ratio=1.1, track_volatility=0.3, lost_volatility=0.1, size_volatility=0.03, threshold=0.3, overlap=0.9, \
                integrate_step=i_s, integrate_threshold=i_t, integrate_volatility=0.8, use_tracking=True, \
                    prohibit_integrate=0.7, size_limit_rate=3, gc=0.01, gc_term=100, gc_success=0.01, visualize=True)
        video.set_out_path(output)
        
        # run test code
        compatible_face, face_area = trimer(video)
        
        write_face_area(area_path, face_area)
        face_area = load_face_area(area_path)
        logger.info('area num : ' + str(len(compatible_face)))
        
        return

        
        # visualize result
        video.reset()
        
        logger.info("draw result for video")
        Visualizer.face_area(face_area, video)
    
if __name__ == "__main__":
    logger = set_logger("TEST-TRIM", "log/test/test-trim.log")
    paths = [
        # ('./data/test/short1.mp4', 'test/trim/out/short1.mp4', 'test/trim/out/short1.area'),
        # ('./data/test/short2.mp4', 'test/trim/out/short2.mp4', 'test/trim/out/short2.area'),
        ('./data/test/test1.mp4', 'test/trim/out/test1.mp4', 'test/trim/out/test1.area'),
        # ('./data/test/test2.mp4', 'test/trim/out/test2.mp4', 'test/trim/out/test2.area'),
        # ('./data/test/test3.mp4', 'test/trim/out/test3.mp4', 'test/trim/out/test3.area'),
        # ('./data/test/test4.mp4', 'test/trim/out/test4.mp4', 'test/trim/out/test4.area')
    ]
    process(logger, paths)