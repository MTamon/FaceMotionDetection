import os
import time
from logging import Logger
from multiprocessing import Process, Queue
from typing import Iterable, List, Tuple

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
            visualize (bool, optional): visualize result as video. Defaults to False.
            result_length (int, optional): result's max step number. memory saving effect. Defaults to 100000.
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

    def __call__(self, paths: list, areas: list) -> List[List[str]]:
        """
        Args:
            paths (list): path list that [(video_path, hp_file_path, visualize_path), ...]\n
            areas (list): trim-area list that [[one_video: {area1}, {area2}, ...], ...]

        Returns:
            List[List[str]]: face-motion list that [[one_video: area1_motion_path, area2_motion_path, ...], ...]
        """

        results = []

        for i, (phase_paths, phase_areas) in enumerate(zip(paths, areas)):
            video_path = phase_paths[0]
            hp_file_path = phase_paths[1]
            visualize_path = None

            if self.visualize:
                if len(phase_paths) < 3:
                    ValueError("visualize-mode needs visualize-path")
                visualize_path = phase_paths[2]

            hp_phase_paths = self.phase(
                video_path, phase_areas, hp_file_path, visualize_path
            )

            results.append(hp_phase_paths)

        return results

    def phase(
        self, input_v: str, areas: List[dict], hp_path: str, output: str = None
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

        self.logger.info("complete estimation process!\n")

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
                    if not result[0]["origin"] is None:
                        frame = Visualizer.head_pose_plotter(frame, result[0])
                    Visualizer.frame_writer(frame, video)

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
        area_info: dict,
        origin: Iterable,
        angle: Iterable,
        landmarks: Iterable,
    ) -> dict:
        """create dictionary for record inference result.

        Args:
            step (int): frame step
            area_info (dict): This dict has 'xmin', 'ymin', 'width', 'height', 'birthtime'.
            origin (Iterable): This iter has two elements (x, y) that face nose position.
            angle (Iterable): This iter has three elements (x, y, z) that face rotation.
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

        activation = step >= area["birthtime"]

        return {
            "step": step,
            "area": area,
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

    def process(
        self, step: int, area: dict, frame: np.ndarray, face_mesh: FaceMesh
    ) -> List[dict]:

        if step < area["birthtime"]:
            return [self.create_dict(step, area, None, None, None)]

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

        img_h, img_w, img_c = image.shape

        results = []

        if recognission.multi_face_landmarks:
            for face_landmarks in recognission.multi_face_landmarks:

                face_3d = []
                face_2d = []

                for idx, lm in enumerate(face_landmarks.landmark):
                    if (
                        idx == 33
                        or idx == 263
                        or idx == 1
                        or idx == 61
                        or idx == 291
                        or idx == 152
                        or idx == 10
                    ):
                        if idx == 1:
                            nose_3d = (lm.x, lm.y, lm.z)
                            continue

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)
                face_2d = face_2d - face_2d.mean(axis=0)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)
                face_3d = face_3d - face_3d.mean(axis=0)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array(
                    [
                        [focal_length, 0, img_h / 2],
                        [0, focal_length, img_w / 2],
                        [0, 0, 1],
                    ]
                )

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(
                    face_3d, face_2d, cam_matrix, dist_matrix
                )

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # inference information dictionary
                head_pose = self.create_dict(
                    step, area, nose_3d, angles, face_landmarks
                )

                results.append(head_pose)

        if results == []:
            return [self.create_dict(step, area, None, None, None)]

        return results

    @staticmethod
    def local2global(
        area: dict, frame_size: Iterable[int], coordinate: Iterable[float]
    ) -> Tuple[float]:
        area_xmin = frame_size[0] * area["xmin"]
        area_ymin = frame_size[1] * area["ymin"]
        area_width = frame_size[0] * area["width"]
        area_height = frame_size[1] * area["height"]

        relative_x = area_width * coordinate[0]
        relative_y = area_height * coordinate[1]

        absolute_x = area_xmin + relative_x
        absolute_y = area_ymin + relative_y

        if len(coordinate) == 3:
            absolute_z = area_width * coordinate[2]
            return (
                absolute_x / frame_size[0],
                absolute_y / frame_size[1],
                absolute_z / frame_size[0],
            )

        return (absolute_x / frame_size[0], absolute_y / frame_size[1])
