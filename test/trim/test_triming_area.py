import _path

import math
from src.trim.triming_area import TrimFace
from src.utils import Video
from src.visualizer import Visualizer
from src.io import write_face_area, load_face_area
from logger_gen import set_logger

def process(logger, paths):
    for input, output, area_path in paths:
        # prepare
        video = Video(input, 'mp4v')
        #i_s = math.ceil(len(video)/10000)
        i_s = 1
        visualize = True
        trimer = TrimFace(logger=logger, min_detection_confidence=0.7, model_selection=1,
            frame_step=1, box_ratio=1.1, 
            track_volatility=0.3, lost_volatility=0.1, size_volatility=0.03, sub_track_volatility=1.0, sub_size_volatility=0.5, threshold=0.3, 
            overlap=0.9, integrate_step=i_s, integrate_volatility=0.4, use_tracking=True, prohibit_integrate=0.7, 
            size_limit_rate=4, gc=0.03, gc_term=100, gc_success=0.1, lost_track=2, visualize=visualize
        )
        
        if visualize:
            video.set_out_path(output)
        
        # run test code
        compatible_face, face_area = trimer(video)
        
        write_face_area(area_path, face_area)
        face_area = load_face_area(area_path)
        logger.info('area num : ' + str(len(compatible_face)))
    
if __name__ == "__main__":
    logger = set_logger("TEST-TRIM", "log/test/test-trim.log")
    paths = [
        ('./data/test/webcame.mp4', 'test/trim/out/webcame.mp4', 'test/trim/out/webcame.area'),
        ('./data/test/midol1s.mp4', 'test/trim/out/midol1s.mp4', 'test/trim/out/midol1s.area'),
        ('./data/test/short1.mp4', 'test/trim/out/short1.mp4', 'test/trim/out/short1.area'),
        ('./data/test/short2.mp4', 'test/trim/out/short2.mp4', 'test/trim/out/short2.area'),
        ('./data/test/test1.mp4', 'test/trim/out/test1.mp4', 'test/trim/out/test1.area'),
        ('./data/test/test2.mp4', 'test/trim/out/test2.mp4', 'test/trim/out/test2.area'),
        ('./data/test/test3.mp4', 'test/trim/out/test3.mp4', 'test/trim/out/test3.area'),
        ('./data/test/test4.mp4', 'test/trim/out/test4.mp4', 'test/trim/out/test4.area')
    ]
    process(logger, paths)