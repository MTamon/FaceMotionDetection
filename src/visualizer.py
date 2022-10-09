from typing import List
from tqdm import tqdm
import mediapipe as mp
import cv2
from src.utils import Video

drawing_utils = mp.solutions.drawing_utils
drawSpec = drawing_utils.DrawingSpec(
    thickness=1,
    circle_radius=1,
    color=(244,244,244)
)

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
    def face_area_window(face_area: List[dict], video: Video, frame, results=None, compatible_ids: list=None):
        frame_width = video.cap_width
        frame_height = video.cap_height
        
        # draw face_area
        for area in face_area:
            frame.flags.writeable = True
            
            pxcel_x_min = int(area["xmin"]*frame_width)
            pxcel_y_min = int(area["ymin"]*frame_height)
            pxcel_x_max = int((area["xmin"]+area["width"])*frame_width)
            pxcel_y_max = int((area["ymin"]+area["height"])*frame_height)
            pt1 = (pxcel_x_min, pxcel_y_min)
            pt2 = (pxcel_x_max, pxcel_y_max)
            clr = (255, 255, 0) if area["prev"] is None else (0, 0, 255)
            cv2.rectangle(frame, pt1, pt2, clr, thickness=1, lineType=cv2.LINE_8, shift=0)
        
        # draw mediapipe's original face detection result
        if not results.detections is None:
            for result in results.detections:
                drawing_utils.draw_detection(frame, result, drawSpec, drawSpec)
                
        # draw compatible id faces
        if not compatible_ids is None:
            for (id, _), face in compatible_ids:
                center = (int(face["xmin"]*frame_width), int(face["ymin"]*frame_height))
                cv2.circle(frame, center=center, radius=3, color=(244,244,244), thickness=1)
            
        video.write(frame)