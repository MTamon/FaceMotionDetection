from logging import Logger
from typing import Dict, Tuple, List
from numpy import append
from tqdm import tqdm

from .face_detection import Detector
from src.utils import Video
from src.visualizer import Visualizer


class TrimFace():
    def __init__(self, min_detection_confidence, model_selection, logger: Logger,
        frame_step=1, box_ratio=1.5, track_volatility=0.3, lost_volatility=0.5, size_volatility=0.3, threshold=0.3, overlap=0.8, 
        integrate_step=1, integrate_volatility=0.3, use_tracking=False, prohibit_integrate=0.7, size_limit_rate=4,
        gc=0.01, gc_term=1000, gc_success=0.05, lost_track=1, visualize=False) -> None:
        
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
        self.integrate_volatility = integrate_volatility
        self.use_tracking = use_tracking
        self.prohibit_integrate = prohibit_integrate
        self.size_limit_rate = size_limit_rate
        self.gc = gc
        self.gc_term = gc_term
        self.gc_success = gc_success
        self.lost_track = lost_track
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
            
            # reset update flag
            for area in face_area:
                area["update"] = False
                
            # face detection
            result_faces, origin = self.detector(frame)
            
            # Mapping area to face
            face_ids = []
            faces = []
            for face in result_faces:
                compatible_face_id = self.judge_face(face_area, face)
                face_ids.append(compatible_face_id)
                faces.append(face)
                
            updates = self.update_table(face_ids, faces)
                
            # update face_area
            for (id, _), face in updates:
                
                if id is None:
                    # add new area
                    area_dict = {"xmin": face["xmin"], "ymin": face["ymin"], "xmax": face["xmin"], "ymax": face["ymin"], "width": face["width"], "height": face["height"], \
                        "width_min": face["width"], "height_min": face["height"], "width_max": face["width"], "height_max": face["height"], \
                            "success": 1, "update": True, "garbage_collect": False, "gc_update": 0, "lost_count": 0, "width_total": face["width"], "height_total": face["height"], \
                                "prev": face, "final_detect": face,}
                    face_area.append(area_dict)
                else:
                    # update existed area
                    face_area[id] =  self.update_area_dict(face_area[id], face)
                    
            # area integration
            if i%self.integrate_step == 0:
                _face_area = self.integrate_area(face_area)
                while len(_face_area) != len(face_area):
                    face_area = _face_area
                    _face_area = self.integrate_area(face_area)
                
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
                    if area["lost_count"] <= self.lost_track:
                        area["lost_count"] += 1
                    else:
                        area["prev"] = None
                    
            # visualize this roop's result
            if self.visualize:
                Visualizer.face_area_window(face_area, video, frame, origin, updates)
                    
        # final area integration
        _face_area = self.integrate_area(face_area)
        while len(_face_area) != len(face_area):
            face_area = _face_area
            _face_area = self.integrate_area(face_area)
            
        # remove under success rate
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
        
        def overlaped(area1: dict, area2: dict) -> dict:
            new_area = area1.copy()
            new_area["xmin"] = min(area1["xmin"], area2["xmin"])
            new_area["ymin"] = min(area1["ymin"], area2["ymin"])
            new_area["xmax"] = max(area1["xmax"], area2["xmax"])
            new_area["ymax"] = max(area1["ymax"], area2["ymax"])
            
            new_area["width_max"] = max(area1["width_max"], area2["width_max"])
            new_area["height_max"] = max(area1["height_max"], area2["height_max"])
            new_area["width_min"] = min(area1["width_min"], area2["width_min"])
            new_area["height_min"] = min(area1["height_min"], area2["height_min"])
            
            new_area["success"] = area1["success"] + area2["success"]
            new_area["prev"] = area1["prev"] if not area1["prev"] is None else area2["prev"]
            new_area["update"] = area1["update"] or area2["update"]
            
            new_area["garbage_collect"] = area1["garbage_collect"] or area2["garbage_collect"]
            new_area["gc_update"] = area1["gc_update"] + area2["gc_update"]
            new_area["lost_count"] = min(area1["lost_count"], area2["lost_count"])
            
            new_area["width_total"] = area1["width_total"] + area2["width_total"]            
            new_area["height_total"] = area1["height_total"] + area2["height_total"]
            
            x_up_lim1 = area1["xmin"] + area1["width"] 
            y_up_lim1 = area1["ymin"] + area1["height"]
            x_up_lim2 = area2["xmin"] + area2["width"] 
            y_up_lim2 = area2["ymin"] + area2["height"]
            x_lim = max(x_up_lim1, x_up_lim2)
            y_lim = max(y_up_lim1, y_up_lim2)
            
            new_area["width"] = x_lim - new_area["xmin"]
            new_area["height"] = y_lim - new_area["ymin"]
            
            new_width = (new_area["width_min"] + new_area["width_max"])/2
            new_height = (new_area["height_min"] + new_area["height_max"])/2
            overlap_flg = True
            if new_width * self.size_limit_rate < new_area["width"] or new_height * self.size_limit_rate < new_area["height"]:
                overlap_flg = False
            if area1["lost_count"] == area2["lost_count"]:
                overlap_flg = False
            
            return [overlap_flg, new_area]
        
        def detect_area_overlapping(area1: dict, area2: dict) -> bool:
            """[overlap, include_relation, x_overlap, y_overlap, overlap_rate[a1_x, a1_y, a2_x, a2_y]]"""
            
            a1_x_low = area1["xmin"]
            a1_x_hig = area1["xmin"] + area1["width"]
            a1_y_low = area1["ymin"]
            a1_y_hig = area1["ymin"] + area1["height"]
            a2_x_low = area2["xmin"]
            a2_x_hig = area2["xmin"] + area2["width"]
            a2_y_low = area2["ymin"]
            a2_y_hig = area2["ymin"] + area2["height"]
            
            x_overlap = 0.
            y_overlap = 0.
            
            if a1_x_low <= a2_x_low <= a1_x_hig:
                x_overlap =  min(a1_x_hig, a2_x_hig) - a2_x_low
            elif a2_x_low <= a1_x_low <= a2_x_hig:
                x_overlap =  min(a2_x_hig, a1_x_hig) - a1_x_low
            else:
                return False
            if a1_y_low <= a2_y_low <= a1_y_hig:
                y_overlap =  min(a1_y_hig, a2_y_hig) - a2_y_low
            elif a2_y_low <= a1_y_low <= a2_y_hig:
                y_overlap =  min(a2_y_hig, a1_y_hig) - a1_y_low
            else:
                return False
                
            overlap_rate = [0., 0., 0., 0.]
            overlap_rate[0] = x_overlap/area1["width"]
            overlap_rate[1] = y_overlap/area1["height"]
            overlap_rate[2] = x_overlap/area2["width"]
            overlap_rate[3] = y_overlap/area2["height"]
            
            # overlap
            if (overlap_rate[0] > self.overlap or overlap_rate[2] > self.overlap) \
                and (overlap_rate[1] > self.overlap or overlap_rate[3] > self.overlap):
                return True
            else:
                return False
        
        new_face_area = []
        skip_id = []
        integrated_flg = False
        for i, area in enumerate(face_area):
            if i in skip_id: # integrated
                continue
            
            for n, _area in enumerate(face_area[i+1:], start=i+1):
                if n in skip_id: # integrated
                    continue
                if area["success"]/self.progress > self.prohibit_integrate and _area["success"]/self.progress > self.prohibit_integrate:
                    continue
                if (not area["prev"] is None) and (not _area["prev"] is None):
                    continue
                
                # size check by size range
                area_width = area["width_max"]
                _area_width = _area["width_max"]
                area_height = area["height_max"]
                _area_height = _area["height_max"]
                
                # size check by size average
                area_width_t = area["width_total"] / area["success"]
                _area_width_t = _area["width_total"] / _area["success"]
                area_height_t = area["height_total"] / area["success"]
                _area_height_t = _area["height_total"] / _area["success"]
                
                if not (abs(1. - area_width_t/_area_width_t) < self.integrate_volatility \
                    and abs(1. - area_height_t/_area_height_t) < self.integrate_volatility):
                    
                    if not (abs(1. - area_width/_area_width) < self.integrate_volatility \
                        and abs(1. - area_height/_area_height) < self.integrate_volatility):
                        continue
                
                overlap = detect_area_overlapping(area, _area)
                
                if overlap:
                    result, new_area = overlaped(area, _area)
                    if not result:
                        continue
                    area = new_area
                    skip_id.append(n)
                    continue
                
            new_face_area.append(area)
            
        return new_face_area
                    
    def update_table(self, face_ids: List[list], faces: List[dict]) -> list:
        face_ids = face_ids.copy()
        
        def update(compatible_face_id: List[tuple], tmp_table: List[list], face: dict, idx: int) -> list:
            if compatible_face_id is None:
                tmp_table.append([(None, None), face, idx])
                return tmp_table
            
            exist_flg = False
            for i, [(id, dist), _, _] in enumerate(tmp_table):
                if compatible_face_id[0][0] == id:
                    exist_flg = True
                    
                    if compatible_face_id[0][1] < dist:
                        former = tmp_table[i][2]
                        former_face = tmp_table[i][1]
                        
                        tmp_table[i][0] = compatible_face_id[0]
                        tmp_table[i][1] = face
                        tmp_table[i][2] = idx
                        
                        if len(face_ids[idx]) > 1:
                            face_ids[idx] = face_ids[idx][1:]
                        else:
                            face_ids[idx] = None
                            
                        tmp_table = update(face_ids[former], tmp_table, former_face, former)
                        
                        break
                    
                    else:
                        if len(face_ids[idx]) > 1:
                            face_ids[idx] = face_ids[idx][1:]
                        else:
                            face_ids[idx] = None
                            
                        tmp_table = update(face_ids[idx], tmp_table, face, idx)
                        
                        break
                        
            if not exist_flg:
                tmp_table.append([compatible_face_id[0], face, idx])
                if len(face_ids[idx]) > 1:
                    face_ids[idx] = face_ids[idx][1:]
                else:
                    face_ids[idx] = None
                    
            return tmp_table
                    
        table = []
        for j in range(len(face_ids)):
            compatible_face_id = face_ids[j]
            face = faces[j]
            
            table = update(compatible_face_id, table, face, j)
            
        _table = [elem[:-1] for elem in table]
        
        return _table
                    
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
        new_area["final_ditect"] = face
        new_area["success"] += 1
        new_area["update"] = True
        new_area["gc_update"] += 1
        new_area["lost_count"] = 0
        
        new_area["width_total"] += face["width"]
        new_area["height_total"] += face["height"]
        
        
        if face["width"] * self.size_limit_rate < new_area["width"] or face["height"] * self.size_limit_rate < new_area["height"]:
            new_area["xmin"] = area["xmin"]
            new_area["ymin"] = area["ymin"]
            new_area["width"] = area["width"]
            new_area["height"] = area["height"]
            return new_area
            
        return new_area
    
    def judge_face(self, face_area: List[dict], face) -> List[tuple]:
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