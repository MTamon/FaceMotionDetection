import _path

from src.visualizer import Visualizer
from logger_gen import set_logger
from src.utils import Video
from src.io import load_face_area

import math


def process(logger, paths):
    for input, output, area_path in paths:
        # prepare
        video = Video(input, 'mp4v')
        face_area = load_face_area(area_path)
        logger.info('all area : ' + str(len(face_area)))
        video.set_out_path(output)
        
        # remove under success rate
        compatible_area = []
        for area in face_area:
            if area["success"]/video.cap_frames > 0.3:
                area["comp"] = True
                compatible_area.append(area)
            else:
                area["comp"] = False
                
        logger.info('area num : ' + str(len(compatible_area)))
        
        # visualize result
        video.reset()
        
        logger.info("draw result for video")
        Visualizer.face_area(face_area, video)

if __name__ == "__main__":
    logger = set_logger("TEST-VISUAL", "log/test/test-visual.log")
    paths = [
        ('./data/test/midol1.mp4', 'test/out/midol1.mp4', 'test/trim/out/midol.area'),
        # ('./data/test/short1.mp4', 'test/trim/out/short1.mp4', 'test/trim/out/short1.area'),
        # ('./data/test/short2.mp4', 'test/trim/out/short2.mp4', 'test/trim/out/short2.area'),
        # ('./data/test/test1.mp4', 'test/trim/out/test1.mp4', 'test/trim/out/test1.area'),
        # ('./data/test/test2.mp4', 'test/trim/out/test2.mp4', 'test/trim/out/test2.area'),
        # ('./data/test/test3.mp4', 'test/trim/out/test3.mp4', 'test/trim/out/test3.area'),
        # ('./data/test/test4.mp4', 'test/trim/out/test4.mp4', 'test/trim/out/test4.area')
    ]
    process(logger, paths)