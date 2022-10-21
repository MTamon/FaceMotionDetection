from logging import Logger
import math
import time
from typing import Tuple, List
from tqdm import tqdm
from multiprocessing import Pool, RLock

from .face_detection import Detector
from src.utils import Video
from src.visualizer import Visualizer

PRIMAL_STEP = 5.0


class TrimFace:
    def __init__(
        self,
        logger: Logger,
        min_detection_confidence=0.7,
        model_selection=1,
        frame_step=1,
        box_ratio=1.1,
        track_volatility=0.3,
        lost_volatility=0.1,
        size_volatility=0.03,
        sub_track_volatility=1.0,
        sub_size_volatility=0.5,
        threshold=0.3,
        overlap=0.9,
        integrate_step=1,
        integrate_volatility=0.4,
        use_tracking=True,
        prohibit_integrate=0.7,
        size_limit_rate=4,
        gc=0.03,
        gc_term=100,
        gc_success=0.1,
        lost_track=2,
        process_num=3,
        visualize=False,
    ) -> None:
        """
        Args:
            logger (Logger):
                Logger instance.
            min_detection_confidence:
                Minimum confidence value ([0.0, 1.0]) for face detection to be considered successful. See details in
                https://solutions.mediapipe.dev/face_detection#min_detection_confidence.
            model_selection:
                0 or 1. 0 to select a short-range model that works best for faces within 2 meters from the camera, and 1 for a full-range model best for faces within 5 meters. See details in
                https://solutions.mediapipe.dev/face_detection#model_selection.
            frame_step (int, optional):
                Frame loading frequency. Defaults to 1.
            box_ratio (float, optional):
                Face-detection bounding-box's rate. Defaults to 1.1.
            track_volatility (float, optional):
                Volatility of displacement during face tracking. Defaults to 0.3.
            lost_volatility (float, optional):
                Volatility of displacement when lost face tracking. Defaults to 0.1.
            size_volatility (float, optional):
                Volatility in face size when face detection. Defaults to 0.03.
            sub_track_volatility (float, optional):
                Volatility of the tracking decision when referring to the last detection position in non-tracking mode, regardless of the period of time lost. Defaults to 1.0.
            sub_size_volatility (float, optional):
                Volatility of the tracking decision when referring to the last detection size in non-tracking mode, regardless of the period of time lost. Defaults to 0.5.
            threshold (float, optional):
                Exclude clippings with low detection rates. Defaults to 0.3.
            overlap (float, optional):
                Integration conditions in duplicate clippings. Defaults to 0.9.
            integrate_step (int, optional):
                Integration running frequency. Defaults to 1.
            integrate_volatility (float, optional):
                Allowable volatility of size features between two clippings when integrating clippings. Defaults to 0.4.
            use_tracking (bool, optional):
                Whether or not to use the face tracking feature. Defaults to True.
            prohibit_integrate (float, optional):
                Threshold to prohibit integration of clippings. Defaults to 0.7.
            size_limit_rate (int, optional):
                Maximum size of the clipping relative to the size of the detected face. Defaults to 4.
            gc (float, optional):
                Success rate thresholds in garbage collect during gc_term. Defaults to 0.03.
            gc_term (int, optional):
                Garbage collection execution cycle. Defaults to 100.
            gc_success (float, optional):
                Success rate thresholds in garbage collect. Defaults to 0.1.
            lost_track (int, optional):
                Number of turns of steps to search the vicinity when face detection is lost. Defaults to 2.
            process_num (int, optional):
                Maximum number of processes in parallel processing. Defaults to 3.
            visualize (bool, optional):
                Visualization options for the analysis process.
                When use this option, processing speed is made be lower. Defaults to False.
        """

        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.logger = logger
        self.frame_step = frame_step
        self.box_ratio = box_ratio

        self.track_volatility = track_volatility
        self.lost_volatility = lost_volatility
        self.size_volatility = size_volatility
        self.sub_track_volatility = sub_track_volatility
        self.sub_size_volatility = sub_size_volatility
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
        self.progress_num = process_num
        self.visualize = visualize

    def __call__(self, paths: List[Tuple[str]]) -> List[List[dict]]:
        """
        Args:
            paths (List[Tuple[str]]): paths have input-video path. in visualize-mode have output-path.

        Returns:
            List: This structure [process][comp or all_area][all]
        """
        process_batch = []
        batch = []
        all_process = len(paths)

        # batch process
        for i, path in enumerate(paths):
            batch.append(path)
            if (i + 1) % self.progress_num == 0 or (i + 1) == all_process:
                process_batch.append(batch)
                batch = []

        results = []
        cur_idx = 0

        for idx, batch in enumerate(process_batch):
            arg_set = []

            dw_sorted = []
            # find largest video file in batch
            for i, proc_path in enumerate(batch):
                video = Video(proc_path[0], "mp4v")
                cell = (i, video.cap_frames)
                dw_sorted = self.insertion_sort(dw_sorted, cell)
            dw_sorted.reverse()

            # generate process
            for n, (i, _) in enumerate(dw_sorted):
                proc_path = batch[i]
                output = None
                if len(proc_path) >= 2:
                    output = proc_path[1]

                self.logger.info(f"process:{i+1} go.")

                arg_set.append([proc_path[0], output, True, n])

            tqdm.set_lock(RLock())
            p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
            results += p.starmap(self.triming_face, arg_set)

            for i in range(len(batch)):
                self.logger.info(
                    f"batch {idx} / process:{i+1} done. -> all area num = {len(results[cur_idx])}"
                )
                cur_idx += 1
            self.logger.info("")

        return results

    def triming_face(
        self, v_path: str, out: str = None, prog: bool = False, idx: int = 0
    ) -> List[dict]:
        video = Video(v_path, "mp4v")
        if self.visualize:
            if out is None:
                raise ValueError(
                    "visualize-mode needs argument 'out', but now it's None."
                )
            video.set_out_path(out)

        # time.sleep(0.2 * idx)

        detector = Detector(
            self.min_detection_confidence, self.model_selection, self.box_ratio
        )

        time.sleep(0.5)

        face_area = []

        video.set_step(self.frame_step)
        iteration = video
        if prog:
            name = video.name.split(".")[0] + " " * 15
            iteration = tqdm(video, desc=name[:15], position=idx)

        self.logger.info("triming face from video ...")
        for i, frame in enumerate(iteration):
            progress = i + 1

            # reset update flag
            for area in face_area:
                area["update"] = False

            # face detection
            result_faces, origin = detector(frame)

            # Mapping area to face
            face_ids = []
            faces = []
            for face in result_faces:
                compatible_face_id = self.judge_face(progress, face_area, face)
                face_ids.append(compatible_face_id)
                faces.append(face)

            updates = self.update_table(face_ids, faces)

            # update face_area
            for (id, _), face in updates:

                if id is None:
                    # add new area
                    area_dict = self.face2area(face, i)
                    face_area.append(area_dict)
                else:
                    # update existed area
                    face_area[id] = self.update_area_dict(progress, face_area[id], face)

            # area integration
            if i % self.integrate_step == 0:
                _face_area = self.integrate_area(progress, face_area)
                while len(_face_area) != len(face_area):
                    face_area = _face_area
                    _face_area = self.integrate_area(progress, face_area)

            # garbage collection
            if progress % self.gc_term == 0:
                face_area = self.garbage_collection(progress, face_area)

            # Areas that were not updated have their past information erased.
            for area in face_area:
                area["lost_count"] += 1
                if not area["update"]:
                    if area["lost_count"] > self.lost_track:
                        area["prev"] = None

            # visualize this roop's result
            if self.visualize:
                Visualizer.face_area_window(face_area, video, frame, origin, updates)

        # final area integration
        _face_area = self.integrate_area(progress, face_area)
        while len(_face_area) != len(face_area):
            face_area = _face_area
            _face_area = self.integrate_area(progress, face_area)

        # remove under success rate
        compatible_area = []
        for area in face_area:
            if area["success"] / video.cap_frames > self.threshold:
                area["comp"] = True
                compatible_area.append(area)
            else:
                area["comp"] = False

        return compatible_area

    def combine_area(self, area1: dict, area2: dict) -> dict:
        comb_area = area1.copy()
        comb_area["birthtime"] = min(area1["birthtime"], area2["birthtime"])

        comb_area["xmin"] = min(area1["xmin"], area2["xmin"])
        comb_area["ymin"] = min(area1["ymin"], area2["ymin"])
        comb_area["xmax"] = max(area1["xmax"], area2["xmax"])
        comb_area["ymax"] = max(area1["ymax"], area2["ymax"])

        comb_area["width_max"] = max(area1["width_max"], area2["width_max"])
        comb_area["height_max"] = max(area1["height_max"], area2["height_max"])
        comb_area["width_min"] = min(area1["width_min"], area2["width_min"])
        comb_area["height_min"] = min(area1["height_min"], area2["height_min"])

        comb_area["success"] = area1["success"] + area2["success"]
        comb_area["update"] = area1["update"] or area2["update"]
        if area1["lost_count"] < area2["lost_count"]:
            comb_area["prev"] = area1["prev"]
            comb_area["final_detect"] = area1["final_detect"]
        else:
            comb_area["prev"] = area2["prev"]
            comb_area["final_detect"] = area2["final_detect"]

        comb_area["garbage_collect"] = (
            area1["garbage_collect"] or area2["garbage_collect"]
        )
        comb_area["gc_update"] = area1["gc_update"] + area2["gc_update"]
        comb_area["lost_count"] = min(area1["lost_count"], area2["lost_count"])

        comb_area["width_total"] = area1["width_total"] + area2["width_total"]
        comb_area["height_total"] = area1["height_total"] + area2["height_total"]

        comb_area["average_wh"]["width"] = (
            comb_area["width_total"] / comb_area["success"]
        )
        comb_area["average_wh"]["height"] = (
            comb_area["height_total"] / comb_area["success"]
        )

        x_up_lim1 = area1["xmin"] + area1["width"]
        y_up_lim1 = area1["ymin"] + area1["height"]
        x_up_lim2 = area2["xmin"] + area2["width"]
        y_up_lim2 = area2["ymin"] + area2["height"]
        x_lim = max(x_up_lim1, x_up_lim2)
        y_lim = max(y_up_lim1, y_up_lim2)

        comb_area["width"] = x_lim - comb_area["xmin"]
        comb_area["height"] = y_lim - comb_area["ymin"]

        return comb_area

    def integrate_area(self, progress: int, face_area: List[dict]) -> List[dict]:
        def overlaped(area1: dict, area2: dict) -> dict:

            new_area = self.combine_area(area1, area2)

            new_width = (new_area["width_min"] + new_area["width_max"]) / 2
            new_height = (new_area["height_min"] + new_area["height_max"]) / 2
            overlap_flg = True
            if (
                new_width * self.size_limit_rate < new_area["width"]
                or new_height * self.size_limit_rate < new_area["height"]
            ):
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

            x_overlap = 0.0
            y_overlap = 0.0

            if a1_x_low <= a2_x_low <= a1_x_hig:
                x_overlap = min(a1_x_hig, a2_x_hig) - a2_x_low
            elif a2_x_low <= a1_x_low <= a2_x_hig:
                x_overlap = min(a2_x_hig, a1_x_hig) - a1_x_low
            else:
                return False
            if a1_y_low <= a2_y_low <= a1_y_hig:
                y_overlap = min(a1_y_hig, a2_y_hig) - a2_y_low
            elif a2_y_low <= a1_y_low <= a2_y_hig:
                y_overlap = min(a2_y_hig, a1_y_hig) - a1_y_low
            else:
                return False

            overlap_rate = [0.0, 0.0, 0.0, 0.0]
            overlap_rate[0] = x_overlap / area1["width"]
            overlap_rate[1] = y_overlap / area1["height"]
            overlap_rate[2] = x_overlap / area2["width"]
            overlap_rate[3] = y_overlap / area2["height"]

            # overlap
            if (overlap_rate[0] > self.overlap or overlap_rate[2] > self.overlap) and (
                overlap_rate[1] > self.overlap or overlap_rate[3] > self.overlap
            ):
                return True
            else:
                return False

        new_face_area = []
        skip_id = []
        for i, area in enumerate(face_area):
            if i in skip_id:  # integrated
                continue

            for n, _area in enumerate(face_area[i + 1 :], start=i + 1):
                if n in skip_id:  # integrated
                    continue
                if (
                    area["success"] / progress > self.prohibit_integrate
                    and _area["success"] / progress > self.prohibit_integrate
                ):
                    continue
                if (area["prev"] is not None) and (_area["prev"] is not None):
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

                if not (
                    abs(1.0 - area_width_t / _area_width_t) < self.integrate_volatility
                    and abs(1.0 - area_height_t / _area_height_t)
                    < self.integrate_volatility
                ):

                    if not (
                        abs(1.0 - area_width / _area_width) < self.integrate_volatility
                        and abs(1.0 - area_height / _area_height)
                        < self.integrate_volatility
                    ):
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

        def update(
            compatible_face_id: List[tuple], tmp_table: List[list], face: dict, idx: int
        ) -> list:
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

                        tmp_table = update(
                            face_ids[former], tmp_table, former_face, former
                        )

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

    def update_area_dict(self, progress: int, area: dict, face) -> dict:

        new_area = self.combine_area(area, self.face2area(face, progress - 1))

        width_limit = face["width"] * self.size_limit_rate
        height_limit = face["height"] * self.size_limit_rate

        if (width_limit < new_area["width"]) or (height_limit < new_area["height"]):
            new_area["xmin"] = area["xmin"]
            new_area["ymin"] = area["ymin"]
            new_area["width"] = area["width"]
            new_area["height"] = area["height"]

        return new_area

    def judge_face(self, progress: int, face_area: List[dict], face) -> List[tuple]:
        """Compare bounding boxes with existing bounding boxes

        Args:
            face_area (list[dict]): List or tuple which have each face area
            face (list[float]): List or tuple which have xmin, width, ymin, height
        """

        def judge_coordinates(face: dict, area: dict, volatility: float):
            radians_x = area["xmax"] - area["xmin"]
            radians_y = area["ymax"] - area["ymin"]

            # for track mode
            if radians_x == 0 or radians_y == 0:
                low_lim_x = area["xmin"] - (area["width"] * volatility) / 2
                high_lim_x = area["xmax"] + (area["width"] * volatility) / 2
                low_lim_y = area["ymin"] - (area["height"] * volatility) / 2
                high_lim_y = area["ymax"] + (area["height"] * volatility) / 2
            # for lost, none-track mode
            else:
                low_lim_x = area["xmin"] - (radians_x * volatility) / 2
                high_lim_x = area["xmax"] + (radians_x * volatility) / 2
                low_lim_y = area["ymin"] - (radians_y * volatility) / 2
                high_lim_y = area["ymax"] + (radians_y * volatility) / 2

            result_x = (face["xmin"] > low_lim_x) and (face["xmin"] < high_lim_x)
            result_y = (face["ymin"] > low_lim_y) and (face["ymin"] < high_lim_y)

            return result_x and result_y

        def judge_size(face: dict, area: dict, volatility: float):
            limit_width_min = area["width_min"] * (1.0 - volatility)
            limit_width_max = area["width_max"] * (1.0 + volatility)
            limit_height_min = area["height_min"] * (1.0 - volatility)
            limit_height_max = area["height_max"] * (1.0 + volatility)

            result_width = (face["width"] > limit_width_min) and (
                face["width"] < limit_width_max
            )
            result_height = (face["height"] > limit_height_min) and (
                face["height"] < limit_height_max
            )

            return result_width and result_height

        def distance_function(
            face: dict, area: dict, candidates_id: list, ref: str, step: int
        ) -> list:
            _candidates = []
            for id in candidates_id:

                area_width = area[id][ref]["width"]
                area_height = area[id][ref]["height"]
                dist_w = abs(math.log(area_width / face["width"]))
                dist_h = abs(math.log(area_height / face["height"]))
                dist = (dist_w**2 + dist_h**2) + PRIMAL_STEP * step
                _candidates = self.insertion_sort(_candidates, (id, dist))

            return _candidates

        compatible_face_id = []

        for id, area in enumerate(face_area):
            final_detect = None
            volatility1 = 0.0
            volatility2 = 0.0
            mode = "track"

            if area["prev"] is None or not self.use_tracking:
                mode = "lost"

                final_detect = self.face2area(area["final_detect"].copy(), progress - 1)
                volatility1 = self.lost_volatility
                volatility2 = self.size_volatility

            else:
                mode = "track"

                volatility1 = self.track_volatility
                volatility2 = self.track_volatility

            # compare with final detect
            if final_detect is not None and mode == "lost":
                if not judge_coordinates(face, final_detect, self.sub_track_volatility):
                    pass
                elif not judge_size(face, final_detect, self.sub_size_volatility):
                    pass
                else:
                    compatible_face_id.append((id, False, True))
                    continue

            # compare coordinates
            if not judge_coordinates(face, area, volatility1):
                continue

            # compare size
            if not judge_size(face, area, volatility2):
                continue

            compatible_face_id.append((id, not area["prev"] is None, False))

        # process for case which more than one candidate is generated
        if len(compatible_face_id) >= 1:
            candidates = []
            final_dtect_candidates = []
            lost_candidates = []

            # primary not "prev" is None.
            for candidate in compatible_face_id:
                if candidate[1]:
                    candidates.append(candidate[0])
                elif candidate[2]:
                    final_dtect_candidates.append(candidate[0])
                else:
                    lost_candidates.append(candidate[0])

            if len(candidates) >= 1:
                # case rest more than one candidates: second, primary size similality
                compatible_face_id = distance_function(
                    face, face_area, candidates, "prev", 0
                )

            elif len(final_dtect_candidates) >= 1:
                # case rest more than one final_detect candidates: second, primary size similality
                compatible_face_id = distance_function(
                    face, face_area, final_dtect_candidates, "final_detect", 1
                )

            elif len(candidates) == 0:
                # case rest no candidates: second, primary size similality
                compatible_face_id = distance_function(
                    face, face_area, lost_candidates, "average_wh", 2
                )

        elif len(compatible_face_id) == 0:
            compatible_face_id = None

        return compatible_face_id

    def face2area(self, face: dict, frame_no) -> dict:
        area_dict = {
            "birthtime": frame_no,
            "xmin": face["xmin"],
            "ymin": face["ymin"],
            "xmax": face["xmin"],
            "ymax": face["ymin"],
            "width": face["width"],
            "height": face["height"],
            "width_min": face["width"],
            "height_min": face["height"],
            "width_max": face["width"],
            "height_max": face["height"],
            "success": 1,
            "update": True,
            "garbage_collect": False,
            "gc_update": 1,
            "lost_count": 0,
            "width_total": face["width"],
            "height_total": face["height"],
            "average_wh": {"width": face["width"], "height": face["height"]},
            "prev": face,
            "final_detect": face,
        }
        return area_dict

    def garbage_collection(self, progress: int, face_area: List[dict]):
        new_face_area = []
        for area in face_area:
            if area["garbage_collect"]:
                if area["success"] / progress < self.gc_success:
                    if area["gc_update"] / self.gc_term < self.gc:
                        continue
            area["garbage_collect"] = True
            area["gc_update"] = 0
            new_face_area.append(area.copy())
        return new_face_area

    def insertion_sort(self, l: list, cell: Tuple[int, float]):
        flg = True
        for idx in range(len(l)):
            if cell[1] > l[idx][1]:
                continue
            else:
                l.insert(idx, cell)
                flg = False
                break

        if flg:
            l.append(cell)

        return l
