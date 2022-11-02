"""This code is for shaping result of HME"""

from logging import Logger
import os
from tqdm import tqdm
from typing import Iterable
import numpy as np

from src.io import load_head_pose
from src.utils import Video, CalcTools
from src.visualizer import Visualizer


class Shaper:
    """This class is for shaping of HME"""

    def __init__(
        self,
        logger: Logger,
        batch_size: int = 5,
        visualize_graph: bool = False,
        visualize_noise: bool = False,
    ):
        self.logger = logger
        self.batch_size = batch_size

        self.visualize_graph = visualize_graph
        self.visualize_noise = visualize_noise

        self.radius = 100
        self.threshold_size = 0.02
        self.threshold_rotate = 2.5
        self.threshold_pos = 0.043
        self.mean_term = 3

    def __call__(self, paths: Iterable[str]):
        """
        Args:
            paths (Iterable[str]): ([input_path, video_path, output_path], ...)
        """
        all_process = len(paths)

        results = []

        for idx, (input_path, video_path, output_path) in enumerate(paths):
            video = Video(video_path, "mp4v")
            resolution = (video.cap_width, video.cap_height)
            fps = video.fps

            self.logger.info(
                f"Progress: {(idx+1)}/{all_process} ... {os.path.basename(input_path)}"
            )

            hme_result = load_head_pose(input_path)
            hme_result = self.to_numpy_landmark(hme_result, resolution)

            init_result = self.init_analysis(hme_result)
            results.append(init_result)

            self.visualize_result(video, hme_result, init_result, output_path)

        return results

    def to_numpy_landmark(
        self, hme_result: np.ndarray, resolution: Iterable[int]
    ) -> np.ndarray:

        all_length = len(hme_result)
        scaler = np.array([resolution[0], resolution[1], resolution[0]])

        for step in tqdm(range(all_length), desc="   (to-numpy)  ", leave=False):
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

    def init_analysis(self, target: np.ndarray) -> np.ndarray:
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

        name = "     (Init)    "
        for step in tqdm(range(all_step), desc=name):
            result_dict = {
                "step": step,
                "countenance": None,
                "rotate": None,
                "centroid": None,
                "ratio": None,
                "grad2": None,
                "gradR2": None,
                "gradZ1": None,
                "noise": False,
            }

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

    def visualize_result(
        self,
        video: Video,
        hme_result: np.ndarray,
        init_result: np.ndarray,
        output_path: str,
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
            )
