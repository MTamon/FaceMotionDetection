import math
import os
import cv2


class Video:
    
    def __init__(self, video_path: str, codec: str = "mp4v") -> None:
        self.cap = cv2.VideoCapture(video_path)
        self.fourcc = cv2.VideoWriter_fourcc(*codec)
        
        self.path = video_path
        self.name = os.path.basename(video_path)
        self.codec = codec
        
        self.cap_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cap_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.cap_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.writer = None
        
        self.step = 1
        
        self.current_idx = 0
        
        self.length = None
        self.__len__()
    
    def __str__(self) -> str:
        return f"all frame : {self.cap_frames}, fps : {round(self.fps, 2)}, time : {round(self.cap_frames/self.fps, 2)}"
    
    def __getitem__(self, idx):
        pos = cv2.CAP_PROP_POS_FRAMES
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        
        return ret, frame
    
    def __len__(self) -> int:
        if self.length is None:
            self.length = math.ceil(self.cap_frames / self.step)
        return self.length
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_idx == self.length:
            raise StopIteration
        
        frame = self.cap.read()[1]
        self.current_idx += 1
        for i in range(self.step - 1):
            self.cap.read()
        return frame
    
    def reset(self):
        self.current_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def info(self):
        return [self.fourcc, self.cap_width, self.cap_height, self.fps, self.cap_frames]
    
    def read(self):
        return self.cap.read()
    
    def set_out_path(self, path: str):
        self.writer = cv2.VideoWriter(
            path, self.fourcc, self.fps, (self.cap_width, self.cap_height)
        )
    
    def write(self, frame):
        self.writer.write(frame)
    
    def set_step(self, step):
        self.step = step
