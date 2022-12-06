"""This code is for Head-Motion-Estimation."""

import os
import time
from logging import Logger
from multiprocessing import Pool
from typing import Iterable, List

import cv2
import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh
from src.io import write_head_pose
from src.utils import Video, batching
from src.visualizer import Visualizer
from tqdm import tqdm


class HeadPoseEstimation:
    def __init__(
        self,
        logger: Logger,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_face=1,
        batch_size=5,
        redo=False,
        single_proc=False,
        visualize=False,
        result_length=1000000,
    ) -> None:
        """
        Args:
            logger (Logger): Logger instance.
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face
                detection to be considered successful. See details in
                https://solutions.mediapipe.dev/face_mesh#min_detection_confidence.
            min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
                face landmarks to be considered tracked successfully. See details in
                https://solutions.mediapipe.dev/face_mesh#min_tracking_confidence.
            max_num_faces: Maximum number of faces to detect. See details in
                https://solutions.mediapipe.dev/face_mesh#max_num_faces.
            batch_size (int, optional):
                Batch size.
            redo (bool, optional):
                Redo process when exist result file. Defaults to False.
            single_proc (bool, optional):
                Running with single-thread.
            visualize (bool, optional):
                visualize result as video. Defaults to False.
            result_length (int, optional):
                result's max step number. memory saving effect. Defaults to 100000.
        """

        self.logger = logger
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_face = max_num_face
        self.batch_size = batch_size
        self.redo = redo
        self.visualize = visualize
        self.result_length = result_length
        self.single_proc = single_proc

        self.detector_args = {
            "static_image_mode": False,
            "max_num_faces": max_num_face,
            "refine_landmarks": True,
            "min_detection_confidence": self.min_detection_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
        }

    def __call__(self, all_phase_args: list) -> List[List[str]]:
        """
        Args:
            all_phase_args: list
                [(video_path, hpe_result_path, area_dict, visualize_path), ...]

        Returns:
            List[List[str]]: face-motion list that [[one_video: area1_motion_path, area2_motion_path, ...], ...]
        """

        results = []
        all_args = []
        hpe_path_order = []

        for idx, phase_args in enumerate(all_phase_args):
            video_path = phase_args[0]
            hp_file_path = phase_args[1]
            phase_area = phase_args[2]
            visualize_path = None

            video_len = len(Video(video_path))

            if self.visualize:
                if len(phase_args) < 4:
                    ValueError("visualize-mode needs visualize-path")
                visualize_path = phase_args[3]

            for i, area in enumerate(phase_area):
                all_args.append(
                    [video_path, area, i, hp_file_path, visualize_path, False]
                )

                hpe_path = self.generate_path(hp_file_path, (0, video_len - 1), i)
                hpe_path_order.append(hpe_path)

        all_args = batching(all_args, self.batch_size)

        ################################
        all_process = len(all_args)

        results = []

        for idx, batch in enumerate(all_args):
            self.logger.info(f" >> Progress: {(idx+1)}/{all_process} << ")

            if not self.single_proc:
                batch[0][5] = True  # tqdm ON
                with Pool(processes=None) as pool:
                    results += pool.starmap(self.phase, batch)
            else:
                for _ba in batch:
                    _ba[5] = True
                    results.append(self.phase(*_ba))

        self.logger.info(" >> DONE. << ")
        #################################

        results = self.match_hp_order(hpe_path_order, results)

        return results

    def match_hp_order(self, order_path, results):
        _results = []
        for _path in order_path:
            for result in results:
                if _path == result:
                    _results.append(result)
                    break
        return _results

    def phase(
        self,
        input_v: str,
        area: dict,
        area_id: int,
        hp_file_path: str,
        output: str = None,
        progress: bool = False,
    ) -> str:

        # wait for tensor-flow message
        time.sleep(area_id * 0.1)

        recognizer = FaceMesh(**self.detector_args)
        video = Video(input_v, "mp4v")

        results = np.array([])
        last_step = 0
        idx = 0

        # When exist results & self.redo == False
        _hp_path = self.generate_path(
            hp_file_path, (last_step, len(video) - 1), area_id
        )
        if os.path.isfile(_hp_path) and not self.redo:
            return _hp_path

        # wait for tensor-flow message
        time.sleep(1)

        if progress:
            if self.visualize:
                if output is None:
                    raise ValueError(
                        "visualize-mode needs argument 'output', but now it's None."
                    )
                video.set_out_path(output)
            name = video.name.split(".")[0] + " " * 15

            for idx, frame in enumerate(tqdm(video, desc=name[:15])):

                result = self.process(idx, area, frame, recognizer)
                result = np.array(result)

                results = np.concatenate((results, result), axis=-1)

                if self.visualize:
                    if result[0]["origin"] is not None:
                        frame = Visualizer.head_pose_plotter(frame, result[0])
                    Visualizer.frame_writer(frame, video)

            if self.visualize:
                video.close_writer()

        else:
            for idx, frame in enumerate(video):

                result = self.process(idx, area, frame, recognizer)
                result = np.array(result)

                results = np.concatenate((results, result), axis=-1)

        hp_path = self.write_result(hp_file_path, results, (last_step, idx), area_id)

        return hp_path

    def write_result(
        self, hp_path: str, results: np.ndarray, term: Iterable[int], area_id: int
    ) -> str:
        path = self.generate_path(hp_path, term, area_id)
        write_head_pose(path, results)

        return path

    def generate_path(self, hp_path: str, term: Iterable[int], area_id: int) -> str:
        hp_base_name = os.path.basename(hp_path)
        hp_dir_name = os.path.dirname(hp_path)

        name, ftype = hp_base_name.split(".")
        name = name + f"_{area_id}_{term[0]}_{term[1]}.{ftype}"
        path = hp_dir_name + "/" + name

        return path

    def create_args(
        self,
        step: int,
        areas: List[dict],
        frame: np.ndarray,
        recognizers: list,
    ):
        args = []
        for area, recognizer in zip(areas, recognizers):
            args.append([step, area, frame, recognizer])

        return args

    def create_dict(
        self,
        step: int,
        img_size: Iterable[int],
        area_info: dict,
        origin: Iterable,
        angle: Iterable,
        landmarks: Iterable,
    ) -> dict:
        """create dictionary for record inference result.

        Args:
            step (int): frame step
            img_size (Iterable[int]): frame resolution. (width x height)
            area_info (dict): This dict has 'xmin', 'ymin', 'width', 'height'.
            origin (Iterable): This iter has two elements (x, y) that face nose position.
            angle (Iterable): This is 3x3 rotation matrix.
            landmarks (Iterable): This iter has face-mesh landmarks [(x, y, z), ...].

        Returns:
            dict: {'area': area_info, 'origin': origin, 'angles': angle, 'landmarks': landmarks}
        """
        area = {}
        area["xmin"] = area_info["xmin"]
        area["ymin"] = area_info["ymin"]
        area["width"] = area_info["width"]
        area["height"] = area_info["height"]

        resolution = (img_size[0], img_size[1])

        return {
            "step": step,
            "area": area,
            "resolution": resolution,
            "origin": origin,
            "angles": angle,
            "landmarks": landmarks,
        }

    def trim_area(self, frame: np.ndarray, area: dict) -> Iterable[int]:
        frame_h, frame_w, _ = frame.shape

        x_lw = area["xmin"]
        x_up = area["xmin"] + area["width"]
        y_lw = area["ymin"]
        y_up = area["ymin"] + area["height"]

        return (
            int(x_lw * frame_w),
            int(x_up * frame_w),
            int(y_lw * frame_h),
            int(y_up * frame_h),
        )

    def calc_R(self, lm, img_w, img_h) -> np.ndarray:
        """
        Calculate rotation matrix.
        Rotate the FaceMesh by R to face the front
        """
        scale_vec = np.array([img_w, img_h, img_w])
        p33 = np.array([lm[33].x, lm[33].y, lm[33].z]) * scale_vec
        p263 = np.array([lm[263].x, lm[263].y, lm[263].z]) * scale_vec
        p152 = np.array([lm[152].x, lm[152].y, lm[152].z]) * scale_vec
        p10 = np.array([lm[10].x, lm[10].y, lm[10].z]) * scale_vec

        _x = p263 - p33
        x = _x / np.linalg.norm(_x)

        _y = p152 - p10
        xy = x * np.dot(x, _y)
        y = _y - xy
        y = y / np.linalg.norm(y)

        z = np.cross(x, y)
        z = z / np.linalg.norm(y)

        R = np.array([x, y, z])

        return R

    def process(
        self, step: int, area: dict, frame: np.ndarray, face_mesh: FaceMesh
    ) -> List[dict]:

        img_h, img_w, _ = frame.shape
        resolution = (img_w, img_h)

        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # triming area
        x_lw, x_up, y_lw, y_up = self.trim_area(frame, area)
        image = image[y_lw:y_up, x_lw:x_up]

        # To improve performance
        image.flags.writeable = False

        # Get the result
        recognission = face_mesh.process(image.copy())

        # To improve performance
        image.flags.writeable = True

        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, _ = image.shape

        results = []

        if recognission.multi_face_landmarks:
            for face_landmarks in recognission.multi_face_landmarks:

                lm = face_landmarks.landmark

                R = self.calc_R(lm, img_w, img_h)

                nose_3d = [lm[1].x, lm[1].y, lm[1].z]

                # inference information dictionary
                head_pose = self.create_dict(
                    step, resolution, area, nose_3d, R.T, face_landmarks
                )

                results.append(head_pose)

        if results == []:
            return [self.create_dict(step, resolution, area, None, None, None)]

        return results
