import cv2


class Video():
    def __init__(self, args) -> None:
        self.cap = cv2.VideoCapture(args.file)
        self.fourcc = cv2.VideoWriter_fourcc(*args.file_codec)
        
        self.cap_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cap_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.cap_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
    def __str__(self) -> str:
        return f'all frame : {self.cap_frames}, fps : {self.fps}, time : {self.cap_frames/self.fps}'
    
    def __getitem__(self, idx):
        pos = cv2.CAP_PROP_POS_FRAMES
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        
        return ret, frame
    
    def __len__(self) -> int:
        return self.cap_frames
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.cap.read()[1]
    
    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def info(self):
        return [self.fourcc, self.cap_width, self.cap_height, self.fps, self.cap_frames]
    
    def read(self):
        return self.cap.read()