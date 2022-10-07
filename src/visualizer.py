from typing import List
from tqdm import tqdm
import cv2
from src.utils import Video

class Visualizer():
    
    @staticmethod
    def face_area(face_area: List[dict], video: Video):
        frame_width = video.cap_width
        frame_height = video.cap_height
        
        for frame in tqdm(video, desc=video.name.split('.')[0]):
            for area in face_area:
                frame.flags.writeable = True
                
                pxcel_x_min = int(area["xmin"]*frame_width)
                pxcel_y_min = int(area["ymin"]*frame_height)
                pxcel_x_max = int((area["xmin"]+area["width"])*frame_width)
                pxcel_y_max = int((area["ymin"]+area["height"])*frame_height)
                pt1 = (pxcel_x_min, pxcel_y_min)
                pt2 = (pxcel_x_max, pxcel_y_max)
                clr = (0, 0, 255) if area["comp"] else (255, 255, 0)
                cv2.rectangle(frame, pt1, pt2, clr, thickness=1, lineType=cv2.LINE_8, shift=0)
                
            video.write(frame)
                
    @staticmethod
    def face_area_window(face_area: List[dict], video: Video, frame):
        frame_width = video.cap_width
        frame_height = video.cap_height
        
        for area in face_area:
            frame.flags.writeable = True
            
            pxcel_x_min = int(area["xmin"]*frame_width)
            pxcel_y_min = int(area["ymin"]*frame_height)
            pxcel_x_max = int((area["xmin"]+area["width"])*frame_width)
            pxcel_y_max = int((area["ymin"]+area["height"])*frame_height)
            pt1 = (pxcel_x_min, pxcel_y_min)
            pt2 = (pxcel_x_max, pxcel_y_max)
            clr = (255, 255, 0)
            cv2.rectangle(frame, pt1, pt2, clr, thickness=1, lineType=cv2.LINE_8, shift=0)
            
        #cv2.imshow(video.name.split('.')[0], frame)
        video.write(frame)