"""This code is for shaping result of HME"""

from logging import Logger
from tqdm import tqdm
from typing import Iterable, List
import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool
import time

from src.io import load_head_pose
from src.utils import Video, CalcTools
from src.visualizer import Visualizer


class Shaper:
    """This class is for shaping of HME"""

    def __init__(
        self,
        logger: Logger,
        batch_size: int = 5,
        threshold_size=0.02,
        threshold_rotate=2.5,
        threshold_pos=0.043,
        visualize_graph: bool = False,
        visualize_noise: bool = False,
    ):
        self.logger = logger
        self.batch_size = batch_size

        self.visualize_graph = visualize_graph
        self.visualize_noise = visualize_noise

        self.radius = 100
        self.threshold_size = threshold_size
        self.threshold_rotate = threshold_rotate
        self.threshold_pos = threshold_pos
        self.mean_term = 3

        self.allow_term = 5

    def __call__(self, paths: Iterable[str]) -> List[str]:
        """
        Args:
            paths (Iterable[str]): ([input_path, video_path, output_path], ...)

        Retern
            results (List[str]): path-list for .sh file
        """

        batches = self.batching(paths)

        all_process = len(batches)
        pool = Pool()

        results = []

        for idx, batch in enumerate(batches):
            self.logger.info(f" >> Progress: {(idx+1)}/{all_process} << ")

            results += pool.starmap(self.phase, batch)

        return results

    def batching(self, paths) -> list:
        batches = []
        batch = []
        max_id = 0
        max_frame = 0

        for idx, (input_path, video_path, output_path) in enumerate(paths):
            video = Video(video_path, "mp4v")
            all_frame = video.cap_frames

            if all_frame > max_frame:
                max_id = idx % self.batch_size
                max_frame = all_frame

            phase_arg_set = [input_path, video_path, output_path, False]
            batch.append(phase_arg_set)

            if (idx + 1) % self.batch_size == 0:
                batch[max_id][3] = True
                batches.append(batch)
                batch = []
                max_id = 0
                max_frame = 0

        return batches

    def phase(
        self,
        input_path: str,
        video_path: str,
        output_path: str,
        tqdm_visual: bool = False,
    ):
        video = Video(video_path, "mp4v")
        resolution = (video.cap_width, video.cap_height)
        fps = video.fps

        time.sleep(1.0)

        hme_result = load_head_pose(input_path)
        hme_result = self.to_numpy_landmark(hme_result, resolution, tqdm_visual)

        init_result = self.init_analysis(hme_result, tqdm_visual)

        self.visualize_result(video, hme_result, init_result, output_path, tqdm_visual)

        return output_path

    def to_numpy_landmark(
        self,
        hme_result: np.ndarray,
        resolution: Iterable[int],
        tqdm_visual: bool = False,
    ) -> np.ndarray:

        all_length = len(hme_result)
        scaler = np.array([resolution[0], resolution[1], resolution[0]])

        if tqdm_visual:
            progress_iterator = tqdm(range(all_length), desc="   (to-numpy)  ")
        else:
            progress_iterator = range(all_length)

        for step in progress_iterator:
            mp_lmarks = hme_result[step]["landmarks"]

            if mp_lmarks is None:
                continue

            facemesh = []
            for landmark in mp_lmarks.landmark:
                landmark = np.array([landmark.x, landmark.y, landmark.z])
                landmark = CalcTools.local2global(
                    hme_result[step]["area"], resolution, landmark
                )
                landmark *= scaler
                facemesh.append(landmark)
            facemesh = np.array(facemesh)

            hme_result[step]["landmarks"] = facemesh

        return hme_result

    def init_analysis(
        self, target: np.ndarray, tqdm_visual: bool = False
    ) -> np.ndarray:
        """For Initialize Analysis befor Main Analysis

        Args:
            target (np.ndarray): result of HME.
            re_calc (np.ndarray): re-calculation HME.
        """
        self.logger.info("Initialize Analysis running ...")

        all_step = len(target)

        prev_centroid = None
        prev_grad1 = None
        prev_R = None
        prev_grad_R = None
        prev_size = None

        results = []

        mean_buf = np.array([None for _ in range(self.mean_term)])

        if tqdm_visual:
            name = "     (Init)    "
            progress_iterator = tqdm(range(all_step), desc=name)
        else:
            progress_iterator = range(all_step)

        for step in progress_iterator:
            result_dict = self.create_dict_result(step)
            result_dict = self.add_dict_process(result_dict)

            facemesh = target[step]["landmarks"]

            if facemesh is None:
                result_dict["noise"] = True
                results.append(result_dict)

                prev_centroid = None
                prev_grad1 = None
                prev_R = None
                prev_grad_R = None
                prev_size = None
                mean_buf[step % self.mean_term] = None
                continue

            mean_buf[step % self.mean_term] = facemesh
            means = []
            for a in mean_buf:
                if a is not None:
                    means.append(a)
            means = np.array(means)
            facemesh = np.sum(means, axis=0) / len(means)

            R = self.calc_R(facemesh)
            result_dict["rotate"] = R

            forward_face, centroid, ratio = self.rotate(facemesh, R)
            result_dict["countenance"] = forward_face
            result_dict["centroid"] = centroid
            result_dict["ratio"] = ratio

            scaler = np.linalg.norm(facemesh[1] - facemesh[10])

            # face size difference
            size = np.linalg.norm(facemesh - centroid, axis=1)
            if prev_size is not None:
                volatilitys = np.log(size) - np.log(prev_size)
                volatility = np.sum(volatilitys) / len(volatilitys)
                result_dict["gradZ1"] = abs(volatility)

                if abs(volatility) > self.threshold_size:
                    result_dict["noise"] = True
            prev_size = size

            # rotation difference (expression by two points ditance on sphere)
            if prev_R is not None:
                base_vec = np.array([self.radius, 0, 0])
                diff_vec = np.dot(np.dot(prev_R, R.T), base_vec)
                cos_theta = np.dot(diff_vec, base_vec) / self.radius**2
                theta = np.arccos(cos_theta)
                grad_R = abs(self.radius * theta)

                if prev_grad_R is not None:
                    gradR2 = abs(grad_R - prev_grad_R)
                    result_dict["gradR2"] = gradR2

                    if gradR2 > self.threshold_rotate:
                        result_dict["noise"] = True

                prev_grad_R = grad_R
            prev_R = R

            # below, it is not speed and acceleration
            if prev_centroid is not None:
                grad1 = (centroid - prev_centroid) / scaler

                if prev_grad1 is not None:
                    grad2 = grad1 - prev_grad1
                    result_dict["grad2"] = grad2

                    if np.linalg.norm(grad2) > self.threshold_pos:
                        result_dict["noise"] = True

                prev_grad1 = grad1
            prev_centroid = centroid

            results.append(result_dict)

        return np.array(results)

    def create_dict_result(self, step) -> dict:
        """Create dictionary for record result"""

        result_dict = {
            "step": step,
            "countenance": None,
            "rotate": None,
            "centroid": None,
            "ratio": None,
        }

        return result_dict

    def add_dict_process(self, result_dict) -> dict:
        """Add new keys to dictionary for processing result"""

        result_dict["grad2"] = None
        result_dict["gradR2"] = None
        result_dict["gradZ1"] = None
        result_dict["noise"] = False
        result_dict["masked"] = False

        return result_dict

    def remove_dict_process(self, reslt_dict) -> dict:
        """Remove keys which is for storing processing result"""

        new_dict = self.create_dict_result(reslt_dict["step"])

        for key in new_dict.keys():
            new_dict[key] = reslt_dict[key]

        return new_dict

    def calc_R(self, facemesh) -> np.ndarray:
        """Calculate rotation matrix"""
        _x = facemesh[263] - facemesh[33]
        x_c = _x / np.linalg.norm(_x)

        _y = facemesh[152] - facemesh[10]
        x_y = x_c * np.dot(x_c, _y)
        y_c = _y - x_y
        y_c = y_c / np.linalg.norm(y_c)

        z_c = np.cross(x_c, y_c)
        z_c = z_c / np.linalg.norm(y_c)

        R = np.array([x_c, y_c, z_c])

        return R

    def rotate(self, facemesh: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Rotate face with rotation-matrix."""
        centroid = facemesh.mean(axis=0)
        facemesh = facemesh - centroid

        new_facemesh = np.dot(R, facemesh.T)

        # normalize face size
        target_size = 200
        p33 = new_facemesh[:, 33]
        p263 = new_facemesh[:, 263]
        length = np.linalg.norm(p33 - p263)
        ratio = target_size / length

        new_facemesh *= ratio

        return (new_facemesh.T, centroid, ratio)

    def remove_unstable_area(self, init_result: List[dict], tqdm_visual: bool = False):
        if tqdm_visual:
            progress_iterator = tqdm(init_result, desc="    (remove)   ")
        else:
            progress_iterator = init_result

        for idx, step_result in enumerate(progress_iterator):
            pass

    def interpolation(self, init_result: List[dict], tqdm_visual: bool = False):
        if tqdm_visual:
            progress_iterator = tqdm(init_result, desc="(Interpolation)")
        else:
            progress_iterator = init_result

        # Interpolation for short-time detection loss
        for idx, step_result in enumerate(progress_iterator):
            pass

    def visualize_result(
        self,
        video: Video,
        hme_result: np.ndarray,
        init_result: np.ndarray,
        output_path: str,
        tqdm_visual: bool = False,
    ):
        if self.visualize_graph:
            out_graph_path = ".".join([output_path.split(".")[0], "png"])
            Visualizer.visualize_grads(
                init_result,
                out_graph_path,
                self.threshold_size,
                self.threshold_rotate,
                self.threshold_pos,
            )
        if self.visualize_noise:
            out_video_path = ".".join([output_path.split(".")[0], "mp4"])
            Visualizer.shape_removed(
                video,
                init_result,
                hme_result,
                self.threshold_size,
                self.threshold_rotate,
                self.threshold_pos,
                out_video_path,
                tqdm_visual,
            )
