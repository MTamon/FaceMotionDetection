"""This code is for shaping result of HME"""

from logging import Logger
import os
from tqdm import tqdm
from typing import Iterable
import numpy as np

from src.io import load_head_pose
from src.utils import Video


class Shaper:
    """This class is for shaping of HME"""

    def __init__(self, logger: Logger, batch_size: int = 5, visualize: bool = False):
        self.logger = logger
        self.batch_size = batch_size
        self.visualize = visualize

        self.r = 100

    def __call__(self, paths: Iterable[str]):
        """
        Args:
            paths (Iterable[str]): ([input_path, video_path, output_path], ...)
        """
        all_process = len(paths)

        for idx, (input_path, video_path, output_path) in enumerate(paths):
            video = Video(video_path, "mp4v")
            resolution = (video.cap_width, video.cap_height)
            fps = video.fps

            self.logger.info(
                f"Progress: {(idx+1)}/{all_process} ... {os.path.basename(input_path)}"
            )

            hme_result = load_head_pose(input_path)

            results = self.init_analysis(hme_result, resolution)

            return results

    def init_analysis(
        self,
        target: np.ndarray,
        resolution: Iterable[int] = None,
    ) -> np.ndarray:
        """For Initialize Analysis befor Main Analysis

        Args:
            target (np.ndarray): result of HME.
            re_calc (np.ndarray): re-calculation HME.
        """
        self.logger.info("Initialize Analysis running ...")

        all_step = len(target)

        scale_vec = np.array([resolution[0], resolution[1], resolution[0]])

        prev_centroid = None
        prev_grad1 = None
        prev_R = None
        prev_gradR = None

        results = []

        name = "     (Init)    "
        for step in tqdm(range(all_step), desc=name):
            result_dict = {
                "countenance": None,
                "rotate": None,
                "centroid": None,
                "ratio": None,
                "grad1": None,
                "grad2": None,
                "gradR1": None,
                "gradR2": None,
            }

            mp_lmarks = target[step]["landmarks"]

            if mp_lmarks is None:
                results.append(result_dict)
                prev_centroid = None
                prev_grad1 = None
                prev_R = None
                prev_gradR = None
                continue

            lms = []
            for lm in mp_lmarks:
                lm = np.array([lm.x, lm.y, lm.z]) * scale_vec
                lms.append(lm)
            lms = np.array(lms)

            R = self.calc_R(lms, *resolution)
            result_dict["rotate"] = R

            forward_face, centroid, ratio = self.rotate(lms, R, *resolution)
            result_dict["countenance"] = forward_face
            result_dict["centroid"] = centroid
            result_dict["ratio"] = ratio

            # rotation difference (expression by two points ditance on sphere)
            if prev_R is not None:
                bas_vec = np.array([self.r, 0, 0])
                cur_vec = np.dot(R.T, bas_vec)
                dif_vec = np.dot(prev_R, cur_vec)
                cos_theta = np.dot(dif_vec, bas_vec) / self.r**2
                theta = np.arccos(cos_theta)
                gradR = abs(self.r * theta)
                result_dict["gradR1"] = gradR
            prev_R = R
            if prev_gradR is not None:
                result_dict["gradR2"] = abs(gradR - prev_gradR)
            prev_gradR = gradR

            # below, it is not speed and acceleration
            if prev_centroid is None:
                prev_centroid = centroid
                results.append(result_dict)
                continue
            grad1 = centroid - prev_centroid
            result_dict["grad1"] = grad1
            prev_centroid = centroid

            if prev_grad1 is None:
                prev_grad1 = grad1
                results.append(result_dict)
                continue
            result_dict["grad2"] = grad1 - prev_grad1
            prev_grad1 = grad1

            results.append(result_dict)

        return np.array(results)

    def calc_R(self, lm, img_w, img_h) -> np.ndarray:
        """Calculate rotation matrix"""
        scale_vec = np.array([img_w, img_h, img_w])
        p33 = np.array([lm[33].x, lm[33].y, lm[33].z]) * scale_vec
        p263 = np.array([lm[263].x, lm[263].y, lm[263].z]) * scale_vec
        p152 = np.array([lm[152].x, lm[152].y, lm[152].z]) * scale_vec
        p10 = np.array([lm[10].x, lm[10].y, lm[10].z]) * scale_vec

        _x = p263 - p33
        x_c = _x / np.linalg.norm(_x)

        _y = p152 - p10
        x_y = x_c * np.dot(x_c, _y)
        y_c = _y - x_y
        y_c = y_c / np.linalg.norm(y_c)

        z_c = np.cross(x_c, y_c)
        z_c = z_c / np.linalg.norm(y_c)

        R = np.array([x_c, y_c, z_c])

        return R

    def rotate(self, lms, R: np.ndarray, img_h, img_w) -> np.ndarray:
        """Rotate face with rotation-matrix."""
        lmarks = np.array([0, 0, 0])
        scale_vec = np.array([img_w, img_h, img_w])
        for lm in lms:
            m = np.array([lm.x, lm.y, lm.z]) * scale_vec
            lmarks = np.vstack((lmarks, m))
        lmarks = lmarks[1:]

        centroid = lmarks.mean(axis=0)
        lmarks = lmarks - centroid

        new_lmarks = np.dot(R, lmarks.T)

        # normalize face size
        target_size = 200
        p33 = new_lmarks[:, 33]
        p263 = new_lmarks[:, 263]
        length = np.linalg.norm(p33 - p263)
        ratio = target_size / length

        new_lmarks *= ratio

        return (new_lmarks.T, centroid, ratio)
