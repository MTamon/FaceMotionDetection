"""This code is for Head-Motion-Estimation."""

import os
import time
from logging import Logger
from multiprocessing import Process, Queue
from typing import Iterable, List

import cv2
import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh
from src.io import write_head_pose
from src.utils import Video
from src.visualizer import Visualizer
from tqdm import tqdm


class HeadPoseEstimation:
    def __init__(
        self,
        logger: Logger,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_face=1,
        visualize=False,
        result_length=100000,
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
            visualize (bool, optional):
                visualize result as video. Defaults to False.
            result_length (int, optional):
                result's max step number. memory saving effect. Defaults to 100000.
        """

        self.logger = logger
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_face = max_num_face
        self.visualize = visualize
        self.result_length = result_length

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
        all_process = len(all_phase_args)

        for idx, phase_args in enumerate(all_phase_args):
            video_path = phase_args[0]
            hp_file_path = phase_args[1]
            phase_area = phase_args[2]
            visualize_path = None

            self.logger.info(
                f"Progress: {(idx+1)}/{all_process} ... {os.path.basename(phase_args[0])}"
            )

            if self.visualize:
                if len(phase_args) < 4:
                    ValueError("visualize-mode needs visualize-path")
                visualize_path = phase_args[3]

            hp_phase_paths = self.phase(
                video_path, hp_file_path, phase_area, visualize_path
            )

            results.append(hp_phase_paths)

        return results

    def phase(
        self, input_v: str, hp_path: str, areas: List[dict], output: str = None
    ) -> List[str]:

        procs_set = []
        arg_set = []

        self.logger.info(Video(input_v, "mp4v"))

        # generate process
        for i, area in enumerate(areas):
            queue_output = Queue()
            arg_set.append((queue_output, input_v, area, i, hp_path, output, i == 0))
            procs_set.append(Process(target=self.apply_face_mesh, args=(arg_set[i])))
            self.logger.info(f"process:{i+1} go.")
            procs_set[i].start()

        hp_paths = []

        for i in range(len(areas)):
            q_out = arg_set[i][0]
            hp_paths.append(q_out.get())
            procs_set[i].join()
            self.logger.info(f"process:{i+1} done.")

        self.logger.info("complete estimation process!")
        self.logger.info("")

        return hp_paths

    def apply_face_mesh(
        self,
        q_out: Queue,
        input_v: str,
        area: dict,
        area_id: int,
        hp_file_path: str,
        output: str = None,
        progress: bool = False,
    ) -> List[str]:

        # wait for tensor-flow message
        time.sleep(area_id * 0.1)

        recognizer = FaceMesh(**self.detector_args)
        video = Video(input_v, "mp4v")

        results = np.array([])
        last_step = 0
        idx = 0

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

        q_out.put(hp_path)

    def write_result(
        self, hp_path: str, results: np.ndarray, term: Iterable[int], area_id: int
    ) -> str:
        hp_base_name = os.path.basename(hp_path)
        hp_dir_name = os.path.dirname(hp_path)

        name, ftype = hp_base_name.split(".")
        name = name + f"_{area_id}_{term[0]}_{term[1]}.{ftype}"
        path = hp_dir_name + "/" + name

        write_head_pose(path, results)

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
            area_info (dict): This dict has 'xmin', 'ymin', 'width', 'height', 'birthtime'.
            origin (Iterable): This iter has two elements (x, y) that face nose position.
            angle (Iterable): This is 3x3 rotation matrix.
            landmarks (Iterable): This iter has face-mesh landmarks [(x, y, z), ...].

        Returns:
            dict: {'area': area_info, 'origin': origin, 'angles': angle, 'landmarks': landmarks, 'activation': }
        """
        area = {}
        area["xmin"] = area_info["xmin"]
        area["ymin"] = area_info["ymin"]
        area["width"] = area_info["width"]
        area["height"] = area_info["height"]
        area["birthtime"] = area_info["birthtime"]

        resolution = (img_size[0], img_size[1])

        activation = step >= area["birthtime"]

        return {
            "step": step,
            "area": area,
            "resolution": resolution,
            "origin": origin,
            "angles": angle,
            "landmarks": landmarks,
            "activation": activation,
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
        """Calculate rotation matrix"""
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

        if step < area["birthtime"]:
            return [self.create_dict(step, resolution, area, None, None, None)]

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
