from logging import Logger
from typing import Dict, Tuple, List
from tqdm import tqdm

from .face_detection import Detector
from src.utils import Video
from src.visualizer import Visualizer


class TrimFace():
    def __init__(self, min_detection_confidence, model_selection, logger: Logger,
        frame_step=1, box_ratio=1.5, track_volatility=0.3, lost_volatility=0.5, size_volatility=0.3, threshold=0.3, overlap=0.8, 
        integrate_step=1, integrate_threshold=100, integrate_volatility=0.3, use_tracking=False, prohibit_integrate=0.7, size_limit_rate=4,
        gc=0.01, gc_term=1000, gc_success=0.05, visualize=False) -> None:
        
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.logger = logger
        self.frame_step = frame_step
        self.box_ratio = box_ratio
        self.track_volatility = track_volatility
        self.lost_volatility = lost_volatility
        self.size_volatility = size_volatility
        self.threshold = threshold
        self.overlap = overlap
        self.integrate_step = integrate_step
        self.integrate_threshold = integrate_threshold
        self.integrate_volatility = integrate_volatility
        self.use_tracking = use_tracking
        self.prohibit_integrate = prohibit_integrate
        self.size_limit_rate = size_limit_rate
        self.gc = gc
        self.gc_term = gc_term
        self.gc_success = gc_success
        self.visualize = visualize
        
        self.progress = 0
        
        self.detector = Detector(
            min_detection_confidence,
            model_selection,
            frame_step,
            box_ratio
        )
        
    def __call__(self, video: Video) -> List[dict]:
        return self.triming_face(video)
        
    def triming_face(self, video: Video) -> List[dict]:
        
        face_area = []
        
        video.set_step(self.frame_step)
        
        self.logger.info("triming face from video ...")
        for i, frame in enumerate(tqdm(video, desc=video.name.split('.')[0])):
            self.progress = i+1
            
            result_faces = self.detector(frame)
            
            # reset update flag
            for area in face_area:
                area["update"] = False
                
            # Mapping area to face
            updates = []
            for face in result_faces:
                compatible_face_id = self.judge_face(face_area, face)
                
                updates = self.update_table(updates, compatible_face_id, face)
                
            # update face_area
            for (id, _), face in updates:
                
                if id is None:
                    # add new area
                    area_dict = {"xmin": face["xmin"], "ymin": face["ymin"], "xmax": face["xmin"], "ymax": face["ymin"], "width": face["width"], "height": face["height"], \
                        "width_min": face["width"], "height_min": face["height"], "width_max": face["width"], "height_max": face["height"], \
                            "success": 1, "prev": face, "update":True, "garbage_collect":False, "gc_update":0}
                    face_area.append(area_dict)
                else:
                    # update existed area
                    face_area[id] =  self.update_area_dict(face_area[id], face)
                    
            # area integration
            if i%self.integrate_step == 0:
                face_area = self.integrate_area(face_area)
                
            # garbage collection
            if self.progress % self.gc_term == 0:
                new_face_area = []
                for area in face_area:
                    if area["garbage_collect"]:
                        if area["success"]/self.progress < self.gc_success:
                            if area["gc_update"]/self.gc_term < self.gc:
                                continue
                    area["garbage_collect"] = True
                    area["gc_update"] = 0
                    new_face_area.append(area.copy())
                face_area = new_face_area
                    
            # Areas that were not updated have their past information erased.
            for area in face_area:
                if not area["update"]:
                    area["prev"] = None
                    
            # visualize this roop's result
            if self.visualize:
                Visualizer.face_area_window(face_area, video, frame)
                    
        # final area integration
        _face_area = self.integrate_area(face_area)
        while len(_face_area) != len(face_area):
            face_area = _face_area
            _face_area = self.integrate_area(face_area)
            
        # remove under 0.5 success rate
        compatible_area = []
        for area in face_area:
            if area["success"]/video.cap_frames > self.threshold:
                area["comp"] = True
                compatible_area.append(area)
            else:
                area["comp"] = False
                
        self.logger.info('complete triming process!')
        self.logger.info(f'all area : {len(face_area)}')
                    
        return [compatible_area, face_area]
                    
    def integrate_area(self, face_area: List[dict]) -> List[dict]:
        def included(area1: dict, area2: dict) -> dict:
            """area1 > area2"""
            new_area = area1.copy()
            new_area["xmax"] = area1["xmax"] if area1["xmax"] > area2["xmax"] else area2["xmax"]
            new_area["ymax"] = area1["ymax"] if area1["ymax"] > area2["ymax"] else area2["ymax"]
            new_area["width_max"] = area1["width_max"] if area1["width_max"] > area2["width_max"] else area2["width_max"]
            new_area["height_max"] = area1["height_max"] if area1["height_max"] > area2["height_max"] else area2["height_max"]
            new_area["width_min"] = area1["width_min"] if area1["width_min"] < area2["width_min"] else area2["width_min"]
            new_area["height_min"] = area1["height_min"] if area1["height_min"] < area2["height_min"] else area2["height_min"]
            new_area["success"] = area1["success"] + area2["success"]
            new_area["prev"] = area1["prev"] if not area1["prev"] is None else area2["prev"]
            new_area["update"] = area1["update"] or area2["update"]
            new_area["garbage_collect"] = area1["garbage_collect"] or area2["garbage_collect"]
            new_area["gc_update"] = area1["gc_update"] + area2["gc_update"]
            
            return new_area
        
        def overlaped(area1: dict, area2: dict) -> dict:
            new_area = area1.copy()
            new_area["xmin"] = area1["xmin"] if area1["xmin"] < area2["xmin"] else area2["xmin"]
            new_area["ymin"] = area1["ymin"] if area1["ymin"] < area2["ymin"] else area2["ymin"]
            new_area["xmax"] = area1["xmax"] if area1["xmax"] > area2["xmax"] else area2["xmax"]
            new_area["ymax"] = area1["ymax"] if area1["ymax"] > area2["ymax"] else area2["ymax"]
            new_area["width_max"] = area1["width_max"] if area1["width_max"] > area2["width_max"] else area2["width_max"]
            new_area["height_max"] = area1["height_max"] if area1["height_max"] > area2["height_max"] else area2["height_max"]
            new_area["width_min"] = area1["width_min"] if area1["width_min"] < area2["width_min"] else area2["width_min"]
            new_area["height_min"] = area1["height_min"] if area1["height_min"] < area2["height_min"] else area2["height_min"]
            new_area["success"] = area1["success"] + area2["success"]
            new_area["prev"] = area1["prev"] if not area1["prev"] is None else area2["prev"]
            new_area["update"] = area1["update"] or area2["update"]
            new_area["garbage_collect"] = area1["garbage_collect"] or area2["garbage_collect"]
            new_area["gc_update"] = area1["gc_update"] + area2["gc_update"]
            
            x_up_lim1 = area1["xmin"]+area1["width"] 
            y_up_lim1 = area1["ymin"]+area1["height"]
            x_up_lim2 = area2["xmin"]+area2["width"] 
            y_up_lim2 = area2["ymin"]+area2["height"]
            x_lim = x_up_lim1 if x_up_lim1 > x_up_lim2 else x_up_lim2
            y_lim = y_up_lim1 if y_up_lim1 > y_up_lim2 else y_up_lim2
            new_area["width"] = x_lim - area1["xmin"]
            new_area["height"] = y_lim - area1["ymin"]
            
            new_width = (new_area["width_min"]+new_area["width_max"])/2
            new_height = (new_area["height_min"]+new_area["height_max"])/2
            overlap_flg = True
            if new_width * self.size_limit_rate < new_area["width"] or new_height * self.size_limit_rate < new_area["height"]:
                overlap_flg = False
            
            return [overlap_flg, new_area]
        
        new_face_area = []
        skip_id = []
        integrated_flg = False
        for i, area in enumerate(face_area):
            if i in skip_id: # integrated
                continue
            if area["success"] <self.integrate_threshold:
                new_face_area.append(area)
                continue
            
            x_up_lim = area["xmin"] + area["width"]
            y_up_lim = area["ymin"] + area["height"]
            
            for n, _area in enumerate(face_area[i+1:], start=i+1):
                if n in skip_id: # integrated
                    continue
                # if _area["success"] <self.integrate_threshold:
                #     continue
                if area["success"]/self.progress > self.prohibit_integrate and _area["success"]/self.progress > self.prohibit_integrate:
                    continue
                if (not area["prev"] is None) and (not _area["prev"] is None):
                    continue
                
                # area_width = area["width_max"]+area["width_min"]
                # _area_width = _area["width_max"]+_area["width_min"]
                # area_height = area["height_max"]+area["height_min"]
                # _area_height = _area["height_max"]+_area["height_min"]
                # if not (abs(1. - area_width/_area_width) < self.integrate_volatility and abs(1. - area_height/_area_height) < self.integrate_volatility):
                #     continue
                
                _x_up_lim = _area["xmin"] + _area["width"]
                _y_up_lim = _area["ymin"] + _area["height"]
                
                # judge included: area <- _area
                if area["xmin"] <= _area["xmin"] and _x_up_lim <= x_up_lim:
                    if area["ymin"] <= _area["ymin"] and _y_up_lim <= y_up_lim:
                        new_face_area.append(included(area, _area))
                        skip_id.append(n)
                        integrated_flg = True
                        break
                        
                # judge included: area -> _area
                elif _area["xmin"] <= area["xmin"] and x_up_lim <= _x_up_lim:
                    if _area["ymin"] <= area["ymin"] and y_up_lim <= _y_up_lim:
                        new_face_area.append(included(_area, area))
                        skip_id.append(n)
                        integrated_flg = True
                        break
                    
                # judge overlapping
                elif area["xmin"] <= _area["xmin"] <= area["xmin"]+area["width"] and area["ymin"] <= _area["ymin"] <= area["ymin"]+area["height"]:
                    over_x = x_up_lim - _area["xmin"]
                    over_y = y_up_lim - _area["ymin"]
                    if over_x > area["width"] * self.overlap or over_x > _area["width"] * self.overlap:
                        if over_y > area["height"] * self.overlap or over_y > _area["height"] * self.overlap:
                            # overlaped
                            result, new_area = overlaped(area, _area)
                            if not result:
                                continue
                            new_face_area.append(new_area)
                            integrated_flg = True
                            skip_id.append(n)
                            break
                    
                elif _area["xmin"] <= area["xmin"] <= _area["xmin"]+_area["width"] and _area["ymin"] <= area["ymin"] <= _area["ymin"]+_area["height"]:
                    over_x = _x_up_lim - area["xmin"]
                    over_y = _y_up_lim - area["ymin"]
                    if over_x > area["width"] * self.overlap or over_x > _area["width"] * self.overlap:
                        if over_y > area["height"] * self.overlap or over_y > _area["height"] * self.overlap:
                            # overlaped
                            result, new_area = overlaped(area, _area)
                            if not result:
                                continue
                            new_face_area.append(new_area)
                            integrated_flg = True
                            skip_id.append(n)
                            break
                        
                elif _area["xmin"] <= area["xmin"] <= _area["xmin"]+_area["width"] and area["ymin"] <= _area["ymin"] <= area["ymin"]+area["height"]:
                    over_x = _x_up_lim - area["xmin"]
                    over_y = y_up_lim - _area["ymin"]
                    if over_x > area["width"] * self.overlap or over_x > _area["width"] * self.overlap:
                        if over_y > area["height"] * self.overlap or over_y > _area["height"] * self.overlap:
                            # overlaped
                            result, new_area = overlaped(area, _area)
                            if not result:
                                continue
                            new_face_area.append(new_area)
                            integrated_flg = True
                            skip_id.append(n)
                            break
                        
                elif area["xmin"] <= _area["xmin"] <= area["xmin"]+area["width"] and _area["ymin"] <= area["ymin"] <= _area["ymin"]+_area["height"]:
                    over_x = x_up_lim - _area["xmin"]
                    over_y = _y_up_lim - area["ymin"]
                    if over_x > area["width"] * self.overlap or over_x > _area["width"] * self.overlap:
                        if over_y > area["height"] * self.overlap or over_y > _area["height"] * self.overlap:
                            # overlaped
                            result, new_area = overlaped(area, _area)
                            if not result:
                                continue
                            new_face_area.append(new_area)
                            integrated_flg = True
                            skip_id.append(n)
                            break
                        
            if not integrated_flg:
                new_face_area.append(area)
            integrated_flg = False
            
        return new_face_area
                    
    def update_table(self, table: list, compatible_face_id: tuple, face: dict) -> list:
        if compatible_face_id is None:
            table.append([(None, None), face])
            return table
        
        exist_flg = False
        for i, [(id, dist), _] in enumerate(table):
            if compatible_face_id[0][0] == id:
                if compatible_face_id[0][1] < dist:
                    table[i][0] = compatible_face_id[0]
                    table[i][1] = face
                exist_flg = True
        if not exist_flg:
            table.append([compatible_face_id[0], face])
            
        return table
                    
    def update_area_dict(self, area: dict, face) -> dict:
        new_area = area.copy()
        if face["xmin"] < area["xmin"]:
            new_area["xmin"] = face["xmin"]
            
            dif = area["xmin"] - face["xmin"]
            new_area["width"] += dif
            
        if face["ymin"] < area["ymin"]:
            new_area["ymin"] = face["ymin"]
            
            dif = area["ymin"] - face["ymin"]
            new_area["height"] += dif
            
        if face["xmin"] > area["xmax"]:
            new_area["xmax"] = face["xmin"]
            
        if face["ymin"] > area["ymax"]:
            new_area["ymax"] = face["ymin"]
            
        if face["xmax"] > new_area["xmin"]+new_area["width"]:
            new_area["width"] = face["xmax"]-new_area["xmin"]
            
        if face["ymax"] > new_area["ymin"]+new_area["height"]:
            new_area["height"] = face["ymax"]-new_area["ymin"]
            
        if face["width"] < area["width_min"]:
            new_area["width_min"] = face["width"]
            
        if face["height"] < area["height_min"]:
            new_area["height_min"] = face["height"]
            
        if face["width"] > area["width_max"]:
            new_area["width_max"] = face["width"]
            
        if face["height"] > area["height_max"]:
            new_area["height_max"] = face["height"]
            
        new_area["prev"] = face
        new_area["success"] += 1
        new_area["update"] = True
        new_area["gc_update"] += 1
        
        
        new_width = (new_area["width_min"]+new_area["width_max"])/2
        new_height = (new_area["height_min"]+new_area["height_max"])/2
        if new_width * self.size_limit_rate < new_area["width"] or new_height * self.size_limit_rate < new_area["height"]:
            return area
            
        return new_area
    
    def judge_face(self, face_area: List[dict], face) -> int:
        """Compare bounding boxes with existing bounding boxes
        
        Args:
            face_area (list[dict]): List or tuple which have each face area
            face (list[float]): List or tuple which have xmin, width, ymin, height
        """
        
        volatility = 0.
        
        compatible_face_id = []
        
        for id, area in enumerate(face_area):
            prev_area = area["prev"]
            
            if prev_area is None or not self.use_tracking:
                volatility = self.lost_volatility
                
                # compare coordinates
                if not self.judge_coordinates(face, area, volatility):
                    continue
                
                # compare size
                if not self.judge_size(face, area, self.size_volatility):
                    continue
                
            else:
                volatility = self.track_volatility
                
                prev_area = prev_area.copy()
                
                # compare coordinates
                prev_area["xmax"] = prev_area["xmin"]
                prev_area["ymax"] = prev_area["ymin"]
                if not self.judge_coordinates(face, prev_area, volatility):
                    continue
                
                # compare size
                prev_area["width_min"] = prev_area["width"]
                prev_area["width_max"] = prev_area["width"]
                prev_area["height_min"] = prev_area["height"]
                prev_area["height_max"] = prev_area["height"]
                
                if not self.judge_size(face, prev_area, volatility):
                    continue
                
            compatible_face_id.append((id, not prev_area is None))
            
        # process for case which more than one candidate is generated
        if len(compatible_face_id) >= 1:
            candidates = []
            lost_candidates = []
            
            # primary not "prev" is None.
            for candidate in compatible_face_id:
                if candidate[1]:
                    candidates.append(candidate[0])
                else:
                    lost_candidates.append(candidate[0])
            
            if len(candidates) >= 1:
                # case rest more than one candidates: second, primary size similality
                _candidates = []
                for id in candidates:
                    dist_x = abs(1.-face_area[id]["prev"]["width"]/face["width"])
                    dist_y = abs(1.-face_area[id]["prev"]["height"]/face["height"])
                    dist = (dist_x**2 + dist_y**2)
                    _candidates = self.insertion_sort(_candidates, (id, dist))
                    
                compatible_face_id = _candidates
                
            elif len(candidates) == 0:
                # case rest no candidates: second, primary size similality
                _candidates = []
                for id in lost_candidates:
                    area = face_area[id]
                    area_width = (area["width_max"] + area["width_min"])/2.
                    area_height = (area["height_max"] + area["height_min"])/2.
                    dist_x = abs(1.-area_width/face["width"])
                    dist_y = abs(1.-area_height/face["height"])
                    dist = (dist_x**2 + dist_y**2)
                    _candidates = self.insertion_sort(_candidates, (id, dist))
                
                compatible_face_id = _candidates
                
        elif len(compatible_face_id) == 0:
            compatible_face_id = None
        
        return compatible_face_id
    
    def judge_coordinates(self, face, area: dict, volatility):
        radians_x = area["xmax"] - area["xmin"]
        radians_y = area["ymax"] - area["ymin"]
        
        # for track mode
        if radians_x == 0 or radians_y == 0:
            low_lim_x = area["xmin"] - (area["width"] * volatility)/2
            high_lim_x = area["xmax"] + (area["width"] * volatility)/2
            low_lim_y = area["ymin"] - (area["height"] * volatility)/2
            high_lim_y = area["ymax"] + (area["height"] * volatility)/2
        # for lost, none-track mode
        else:
            low_lim_x = area["xmin"] - (radians_x * volatility)/2
            high_lim_x = area["xmax"] + (radians_x * volatility)/2
            low_lim_y = area["ymin"] - (radians_y * volatility)/2
            high_lim_y = area["ymax"] + (radians_y * volatility)/2
        
        result_x = (face["xmin"] > low_lim_x) and (face["xmin"] < high_lim_x)
        result_y = (face["ymin"] > low_lim_y) and (face["ymin"] < high_lim_y)
                
        return result_x and result_y
    
    def judge_size(self, face, area, volatility):
        result_width = \
            (face["width"] > area["width_min"]*(1. - volatility)) \
                and (face["width"] < area["width_max"]*(1. + volatility))
        result_height = \
            (face["height"] > area["height_min"]*(1. - volatility)) \
                and (face["height"] < area["height_max"]*(1. + volatility))
                
        return result_width and result_height
    
    def insertion_sort(self, l: list, cell: Tuple[int, float]):
        flg = True
        for idx in range(len(l)):
            if cell[1] > l[idx][1]:
                continue
            else:
                l.insert(idx, cell)
                flg = False
                
        if flg:
            l.append(cell)
            
        return l