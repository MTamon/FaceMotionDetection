"""This code is for shaping result of HME"""

from logging import Logger
import os
import shutil
from tqdm import tqdm
from typing import Dict, Iterable, List, Tuple
import numpy as np
from numpy import ndarray
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
        visualize_interpolation: bool = False,
        visualize_all: bool = False,
    ):
        self.logger = logger
        self.batch_size = batch_size

        self.visualize_graph = visualize_graph
        self.visualize_noise = visualize_noise
        self.visualize_interpolation = visualize_interpolation
        self.visualize_all = visualize_all

        self.radius = 100
        self.threshold_size = threshold_size
        self.threshold_rotate = threshold_rotate
        self.threshold_pos = threshold_pos
        self.mean_term = 3

        self.threshold_noise = 0.3
        self.inspection_range = 20
        self.consective_scs = 5
        self.eject_term = 3

        self.order = 7
        self.enhancement = True
        self.interp_margin = 4

        self.ex_cond_nois_len = 10
        self.ex_cond_msk_len = 20

        self.noise_subtract = 0.2
        self.mask_subtract = 0.05

        self.all_weight_mode = True
        self.enhance_end_weight = 300

        self.rotate_limit = 90  # 360 degree base

        self.warnings()

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
            # for element in batch:
            #     self.phase(*element)
            #     break

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
        # fps = video.fps

        time.sleep(0.2)

        # get process name
        process_group = os.path.basename(input_path).split("_")
        if len(process_group) > 2:
            process_name = f"{process_group[0]}_{process_group[1]}"
        else:
            process_name = f"{process_group}_UNK"

        # get visualize path
        _out_dir = os.path.dirname(output_path)
        out_dir = os.path.join(_out_dir, process_name)
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)
        visual_path = os.path.join(out_dir, process_name)

        hme_result = load_head_pose(input_path)
        hme_result = self.to_numpy_landmark(hme_result, resolution, tqdm_visual)

        init_result = self.init_analysis(hme_result, tqdm_visual)

        stable_result = self.remove_unstable_area(init_result, tqdm_visual)

        data_info = self.collect_data_limits()
        interpolation_result = self.interpolation(
            stable_result, data_info, tqdm_visual, visual_path
        )

        self.visualize_result(
            video, hme_result, init_result, stable_result, output_path, tqdm_visual
        )

        return output_path

    def to_numpy_landmark(
        self,
        hme_result: ndarray,
        resolution: Iterable[int],
        tqdm_visual: bool = False,
    ) -> ndarray:

        all_length = len(hme_result)
        scaler = np.array([resolution[0], resolution[1], resolution[0]])

        for i in range(all_length):
            if hme_result[i]["landmarks"] is None:
                continue

            if isinstance(hme_result[i]["landmarks"], ndarray):
                return hme_result

            break

        if tqdm_visual:
            progress_iterator = tqdm(range(all_length), desc="      to-numpy ")
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

    def init_analysis(self, target: ndarray, tqdm_visual: bool = False) -> ndarray:
        """For Initialize Analysis befor Main Analysis

        Args:
            target (ndarray): result of HME.
            re_calc (ndarray): re-calculation HME.
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
            name = "          Init "
            progress_iterator = tqdm(range(all_step), desc=name)
        else:
            progress_iterator = range(all_step)

        for step in progress_iterator:
            result_dict = self.create_dict_result(step)
            result_dict = self.add_dict_process(result_dict)

            facemesh = target[step]["landmarks"]

            if facemesh is None:
                result_dict["noise"] = True
                result_dict["_noise"] = True
                result_dict["noise_type"] = "n"
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
            means = np.stack(means)
            facemesh = np.mean(means, axis=0)

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
                    result_dict["_noise"] = True
                    result_dict["noise_type"] += "s"
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
                        result_dict["_noise"] = True
                        result_dict["noise_type"] += "r"

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
                        result_dict["_noise"] = True
                        result_dict["noise_type"] += "c"

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
        result_dict["noise_type"] = ""
        result_dict["masked"] = False
        result_dict["_masked"] = False
        result_dict["_noise"] = False
        result_dict["ignore"] = False

        return result_dict

    def remove_dict_process(self, reslt_dict) -> dict:
        """Remove keys which is for storing processing result"""

        new_dict = self.create_dict_result(reslt_dict["step"])

        for key in new_dict.keys():
            new_dict[key] = reslt_dict[key]

        return new_dict

    def calc_R(self, facemesh) -> ndarray:
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

    def rotate(self, facemesh: ndarray, R: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
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

    def remove_unstable_area(
        self, init_result: Iterable[dict], tqdm_visual: bool = False
    ) -> ndarray:
        if tqdm_visual:
            progress_iterator = tqdm(init_result, desc="        remove ")
        else:
            progress_iterator = init_result

        noise_info = [r["noise"] for r in init_result]

        for idx, step_result in enumerate(progress_iterator):
            if not step_result["noise"]:
                continue

            if step_result["masked"]:
                continue

            serch_area = noise_info[idx:]

            noise_frame = sum(serch_area[: self.inspection_range])

            tail = idx
            count_scs = 0
            count_fil = 0
            last_count_scs = 0
            last_count_fil = 0
            unstable_area_flg = (
                noise_frame / self.inspection_range > self.threshold_noise
            )
            for inspct_idx, noised in enumerate(serch_area):
                cur_idx = inspct_idx + idx
                if noised:
                    tail = cur_idx

                    last_count_scs = count_scs
                    count_scs = 0
                    count_fil += 1
                else:
                    count_scs += 1
                    last_count_fil = count_fil
                    count_fil = 0

                if unstable_area_flg:
                    if count_scs >= self.consective_scs:
                        break
                    # ex: noise_info[.., F, F, F, T, T, F, F, F, ...]
                    elif (
                        count_scs >= self.eject_term
                        and last_count_scs >= self.eject_term
                    ):
                        if last_count_scs + count_scs > last_count_fil:
                            break
                else:
                    if count_scs >= self.eject_term:
                        break

            for mask_idx in range(idx, tail + 1):
                init_result[mask_idx]["masked"] = True
                init_result[mask_idx]["_masked"] = True

        debug_window = []
        for i, stres in enumerate(init_result):
            debug_window.append([i, stres["noise"], stres["masked"]])

        return init_result

    def interpolation(
        self,
        stable_result: List[dict],
        data_info: dict,
        tqdm_visual: bool = False,
        name: str = None,
    ) -> Tuple[ndarray]:
        all_step = len(stable_result)

        def get_masked_period(start_masked_idx: int):

            period = 0
            consective = [0]

            for msk_idx in range(start_masked_idx, all_step):
                step_result = stable_result[msk_idx]

                if step_result["masked"]:
                    period += 1
                    consective[-1] += 1
                else:
                    break

                if not step_result["noise"] and self.enhancement:
                    consective[-1] -= 1
                    consective.append(0)

            return period, max(consective)

        if tqdm_visual:
            progress_iterator = tqdm(stable_result, desc=" Interpolation ")
        else:
            progress_iterator = stable_result

        masked_period = 0

        # Interpolation for short-time detection loss
        for idx, step_result in enumerate(progress_iterator):
            if masked_period > 0:
                masked_period -= 1
                continue

            if not step_result["masked"]:
                continue

            masked_period, max_consect_len = get_masked_period(idx)

            # If the maximum interpolatable period is exceeded, the data is discarded.
            if (
                max_consect_len > self.ex_cond_nois_len
                and masked_period > self.ex_cond_msk_len
            ):
                for i in range(masked_period):
                    stable_result[idx + i]["countenance"] = None
                    stable_result[idx + i]["rotate"] = None
                    stable_result[idx + i]["centroid"] = None
                    stable_result[idx + i]["ratio"] = None
                    stable_result[idx + i]["ignore"] = True
                continue
            # If a masked area occurs immediately after the start, the data is discarded.
            if idx < self.interp_margin:
                for i in range(idx + masked_period):
                    stable_result[i]["countenance"] = None
                    stable_result[i]["rotate"] = None
                    stable_result[i]["centroid"] = None
                    stable_result[i]["ratio"] = None
                    stable_result[i]["ignore"] = True

                continue

            # Condition for masked data within interpolation margin.(Interpolated in foreground)
            if not self.all_weight_mode:
                mask_b = np.full(self.interp_margin, False)
                for i in range(self.interp_margin):
                    _i = masked_period + i
                    if stable_result[idx + _i]["_masked"]:
                        mask_b[_i - masked_period] = True
                if sum(mask_b) != 0:
                    _noised_period = 0
                    _un_noised_period_f = 0
                    _un_noised_period_b = 0
                    for i in range(masked_period - 1, -1, -1):
                        if stable_result[idx + i]["_noise"]:
                            _noised_period += 1
                        else:
                            break
                    for i in range(self.interp_margin):
                        _i = (masked_period - 1) - _noised_period - i
                        if idx + _i < 0:
                            break
                        if not stable_result[idx + _i]["_noise"]:
                            _un_noised_period_f += 1
                        else:
                            break
                    for i in range(self.interp_margin):
                        _i = masked_period + i
                        if not stable_result[idx + _i]["_noise"]:
                            _un_noised_period_b += 1
                        else:
                            break
                    _idx = idx + masked_period - _noised_period
                    stable_result = self._interpolation(
                        stable_result,
                        data_info,
                        _idx,
                        _noised_period,
                        _un_noised_period_f,
                        _un_noised_period_b,
                        name,
                    )
                    masked_period -= _noised_period - _un_noised_period_f

            stable_result = self._interpolation(
                stable_result,
                data_info,
                idx,
                masked_period,
                self.interp_margin,
                self.interp_margin,
                name,
            )

        return stable_result

    def _interpolation(
        self,
        stable_result: ndarray,
        data_info: dict,
        idx: int,
        masked_period: int,
        margin_f: int,
        margin_b: int,
        name: str = None,
    ) -> ndarray:
        centroids = []
        rotates = []
        countenances = []
        ratios = []
        axis_time = []
        linear_time = []

        interp_range = masked_period + (margin_f + margin_b)
        polynomial_mask = np.full(interp_range, True)  # Indicates interpolation target
        linear_mask = np.full(
            masked_period + 2, True
        )  # Indicates linear-interpolation target

        interp_weight = []
        much_array = np.full(interp_range, -1)

        for i in range(-margin_f, masked_period + margin_b):
            centroid = stable_result[idx + i]["centroid"]
            rotate = stable_result[idx + i]["rotate"]

            countenance = stable_result[idx + i]["countenance"]
            ratio = stable_result[idx + i]["ratio"]

            if countenance is None:
                continue

            if self.all_weight_mode:

                rotates.append(rotate)
                centroids.append(centroid)
                axis_time.append(idx + i)

                much_array[i + margin_f] = len(interp_weight)
                interp_weight.append(1.0)

                if stable_result[idx + i]["_masked"]:
                    interp_weight[-1] -= self.mask_subtract
                if stable_result[idx + i]["noise"]:
                    interp_weight[-1] -= self.noise_subtract
                if i == -1 or i == masked_period:
                    interp_weight[-1] *= self.enhance_end_weight

                if not stable_result[idx + i]["_masked"]:
                    polynomial_mask[i + margin_f] = False

            else:
                if (
                    not stable_result[idx + i]["_masked"]
                    or self.enhancement
                    and not stable_result[idx + i]["noise"]
                ):
                    rotates.append(rotate)
                    centroids.append(centroid)
                    axis_time.append(idx + i)

                    much_array[i + margin_f] = len(interp_weight)
                    interp_weight.append(1.0)

                    # Enhancement points are also interpolated.
                    if self.enhancement and not stable_result[idx + i]["noise"]:
                        interp_weight[-1] -= self.mask_subtract
                    if not stable_result[idx + i]["_masked"]:
                        polynomial_mask[i + margin_f] = False
                    if i == -1 or i == masked_period:
                        interp_weight[-1] *= self.enhance_end_weight

            # if -1 <= i <= masked_period:
            if i in (-1, masked_period):
                ratios.append(ratio)
                countenances.append(countenance)
                linear_time.append(idx + i)
                linear_mask[i + 1] = False

        countenances = np.stack(countenances)
        rotates = np.stack(rotates)
        centroids = np.stack(centroids)
        ratios = np.stack(ratios)
        axis_time = np.array(axis_time)

        # masked_period -= 1 # the masked-area's next step must be "masked" == False

        # interpolation for centroid
        _centroids = self.complex_interpolation(
            centroids.T,
            axis_time,
            interp_weight,
            polynomial_mask,
            data_info["cent"]["up_lim"],
            data_info["cent"]["low_lim"],
            visualize=self.visualize_interpolation,
            visualize_tag=f"{name}_centrd",
            stable_result=stable_result,
        )
        _centroids = _centroids.T

        # interpolation for rotation
        angles = CalcTools.matrix_to_angles(rotates)
        _angles = self.complex_interpolation(
            angles.T,
            axis_time,
            interp_weight,
            polynomial_mask,
            data_info["angle"]["up_lim"],
            data_info["angle"]["low_lim"],
            visualize=self.visualize_interpolation,
            visualize_tag=f"{name}_rotate",
            stable_result=stable_result,
        )
        _angles = _angles.T
        _rotates = CalcTools.angles_to_matrix(_angles)

        # interpolation for countenance
        _countenances = self.linear_interpolation(countenances, linear_time)

        # interpolation for ratio
        _ratios = self.linear_interpolation(ratios, linear_time)

        # _countenances = self.complex_interpolation(
        #     countenances.T,
        #     linear_time,
        #     np.array([1.0, 1.0]),
        #     linear_mask,
        #     sp_order=1,
        # )
        # _countenances = _countenances.T

        # _ratios = self.complex_interpolation(
        #     ratios,
        #     linear_time,
        #     np.array([1.0, 1.0]),
        #     linear_mask,
        #     sp_order=1,
        # )

        for i in range(-margin_f, masked_period + margin_b):
            _idx = idx + i
            _i = i + margin_f

            if polynomial_mask[_i]:

                stable_result[_idx]["centroid"] = _centroids[_i]
                stable_result[_idx]["rotate"] = _rotates[_i]

                if 0 <= i < masked_period:
                    stable_result[_idx]["_masked"] = False
                    stable_result[_idx]["_noise"] = False

                    stable_result[_idx]["countenance"] = _countenances[i + 1]
                    stable_result[_idx]["ratio"] = _ratios[i + 1]

        return stable_result

    def complex_interpolation(
        self,
        x: ndarray,
        t: ndarray,
        w: ndarray,
        mask: ndarray,
        up_limit: Iterable[float] = None,
        low_limit: Iterable[float] = None,
        sp_order: int = None,
        visualize: bool = False,
        visualize_tag: str = None,
        stable_result: ndarray = None,
    ) -> ndarray:
        """interpolation for each datas earch axis.

        Args:
            x (ndarray): interpolation as x(t). Allow the form: x = [x_1(t), x_2(t), ...]
            t (ndarray): this is independence valiable
            w (ndarray): this is numpy.polyfit()'s w. This must be same length of x.
            mask (ndarray[bool]): This indicates whether interpolation should be done.
            up_limit (List[float]): uper limit value of x. This size 1 or len(x). Defaults to None.
            low_limit (List[float]): lower limit value of x. This size 1 or len(x). Defaults to None.
            sp_order (int, optional): Specifies the order of polynomial interpolation. Defaults to None.

        Returns:
            ndarray: time is interpolated every step.
        """

        def recursive(r_x: ndarray) -> ndarray:
            res = []
            if r_x.ndim != 1:
                for p_x in r_x:
                    res.append(recursive(p_x))
                res = np.stack(res)
                return res

            interp_info = np.polyfit(t, r_x, used_order, full=True, w=w)
            func_interp = np.poly1d(interp_info[0])
            _x = func_interp(all_time)

            if visualize:
                curve_x.append(func_interp(graph_time))

            return _x

        used_order = self.order
        if sp_order is not None:
            used_order = sp_order

        start_idx = t[0]
        finis_idx = t[0] + len(mask)

        if visualize:
            graph_time = np.arange(start=start_idx, stop=finis_idx, step=1e-2)
            curve_x = []

        all_time = np.arange(start=start_idx, stop=finis_idx, step=1)
        _x = recursive(x)

        # Correction process when limit values are exceeded.
        if up_limit is not None:
            up_limit = np.array(up_limit).reshape((-1, 1))
            up_lim_inf = np.logical_and((_x > up_limit), mask)
            if np.sum(up_lim_inf) > 0:

                # _x[up_lim_inf] = up_limit <- same works
                c_x = up_lim_inf.astype(np.float64)
                _c_x = 1 - c_x
                _x = _x * _c_x + up_limit * c_x

        if low_limit is not None:
            low_limit = np.array(low_limit).reshape((-1, 1))
            low_lim_inf = np.logical_and(_x < low_limit, mask)
            if np.sum(low_lim_inf) > 0:

                # _x[low_lim_inf] = low_limit <- same works
                c_x = low_lim_inf.astype(np.float64)
                _c_x = 1 - c_x
                _x = _x * _c_x + low_limit * c_x

        # Undo non-interpolated area
        _x_shape = _x.shape
        _x = _x.reshape((-1, _x_shape[-1]))
        x_shape = x.shape
        x = x.reshape((-1, x_shape[-1]))
        for i, _idx in enumerate(t):
            if not mask[_idx - t[0]]:
                _x[:, _idx - t[0]] = x[:, i]
        _x = _x.reshape(_x_shape)
        x = x.reshape(x_shape)

        if visualize:
            if _x.ndim != 2:
                self.logger.warn("Failed to visualize-interpolation")
                return _x
            # sr_sele1 = []
            org_mask = []
            for _t in t:
                if stable_result[_t]["noise"]:
                    org_mask.append("noised")
                elif stable_result[_t]["masked"]:
                    org_mask.append("masked")
                else:
                    org_mask.append("normal")
                # sr_sele1.append(stable_result[_t])
            mod_mask = []
            for _t in all_time:
                if stable_result[_t]["noise"]:
                    mod_mask.append("noised")
                elif stable_result[_t]["masked"]:
                    mod_mask.append("masked")
                else:
                    mod_mask.append("normal")

            Visualizer.interpolation(
                t,
                x,
                org_mask,
                all_time,
                _x,
                mod_mask,
                curve_x,
                graph_time,
                up_limit,
                low_limit,
                f"{visualize_tag}_{start_idx}_{finis_idx}.png",
            )

        return _x

    def linear_interpolation(self, lms: ndarray, linear_time: ndarray):
        all_time = np.arange(start=linear_time[0], stop=linear_time[1] + 1, step=1)

        c_shape = lms.shape
        lms = lms.reshape((2, -1))

        coef_1 = (lms[1] - lms[0]) / (linear_time[1] - linear_time[0])
        coef_0 = lms[0] - coef_1 * linear_time[0]

        res = np.dot(
            all_time.reshape((-1, 1)), coef_1.reshape((1, -1))
        ) + coef_0.reshape((1, -1))

        new_shape = (len(all_time), *c_shape[1:])
        res = res.reshape(new_shape)

        return res

    def collect_data_limits(self, area=None) -> Dict[str, Dict[str, ndarray]]:
        data_limits = {"cent": {}, "angle": {}}

        defo_up = np.array([1280, 720, 1280])
        defo_low = np.array([0, 0, 0])

        data_limits["cent"]["up_lim"] = defo_up
        data_limits["cent"]["low_lim"] = defo_low

        rotate_h_lim = np.full(3, self.rotate_limit)
        rotate_l_lim = np.full(3, -self.rotate_limit)
        data_limits["angle"]["up_lim"] = rotate_h_lim
        data_limits["angle"]["low_lim"] = rotate_l_lim

        return data_limits

    def visualize_result(
        self,
        video: Video,
        hme_result: ndarray,
        init_result: ndarray,
        stable_result: ndarray,
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
        if self.visualize_all:
            out_video_path = ".".join([output_path.split(".")[0] + "ALL", "mp4"])
            Visualizer.shape_result(video, stable_result, out_video_path, tqdm_visual)

    def warnings(self):
        if self.interp_margin > self.consective_scs:
            self.logger.warn(
                "It is recommended that 'consective_scs' be larger than 'interp_margin'."
            )

        lim_noise = int(self.threshold_noise * self.inspection_range * 0.5)
        max_noise = int(self.inspection_range / self.eject_term)
        def_noise = max_noise - lim_noise
        rest_margin = def_noise * self.eject_term
        if self.interp_margin > rest_margin:
            self.logger.warn("There is a risk of interpolation failure.")
