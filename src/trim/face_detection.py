from numpy import ndarray

import mediapipe as mp
import cv2

mp_module = mp.solutions.face_detection


class Detector():
    
    def __init__(self, min_detection_confidence, model_selection, box_ratio=1.5) -> None:
        
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.box_ratio = box_ratio
        
        self.detector = mp_module.FaceDetection(
            min_detection_confidence=self.min_detection_confidence,
            model_selection=self.model_selection
        )
    
    def __call__(self, frame: ndarray) -> list:
        return self.detection(frame)
    
    def detection(self, frame: ndarray) -> list:
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # To improve performance
        imgRGB.flags.writeable = False
        
        result_ditection = self.detector.process((imgRGB))
        if not result_ditection.detections:
            return [], result_ditection
        
        # To improve performance
        imgRGB.flags.writeable = True
        
        bboxes = []
        for detection in result_ditection.detections:
            box_xmin = detection.location_data.relative_bounding_box.xmin
            box_ymin = detection.location_data.relative_bounding_box.ymin
            box_width = detection.location_data.relative_bounding_box.width
            box_height = detection.location_data.relative_bounding_box.height
            
            xmin = box_xmin - box_width * ((self.box_ratio - 1.) / 2)
            width = box_width * self.box_ratio
            xmax = xmin + width
            ymin = box_ymin - box_height * ((self.box_ratio - 1.) / 2)
            height = box_height * self.box_ratio
            ymax = ymin + height
            
            xmin = xmin if xmin > 0 else 0.
            ymin = ymin if ymin > 0 else 0.
            xmax = xmax if xmax < 1 else 1.
            ymax = ymax if ymax < 1 else 1.
            width = width if width < 1 else 1.
            height = height if height < 1 else 1.
            
            bboxes.append({
                "xmin": xmin,
                "xmax": xmax,
                "width": width,
                "ymin": ymin,
                "ymax": ymax,
                "height": height
            })
        
        return bboxes, result_ditection