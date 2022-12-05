"""Extraction Phase for face motion data."""

from argparse import Namespace
from logging import Logger
import os
from .trim.triming_area import TrimFace
from .face_mesh.face_mesh import HeadPoseEstimation
from .io import load_face_area
from .utils import Loging_MSG


class Extraction:
    def __init__(self, logger: Logger, args: Namespace) -> None:
        trim_args = self.get_arg_trim(logger, args)
        hpe_args = self.get_hpe_args(logger, args)

        self.trimer = TrimFace(**trim_args)
        self.hpe = HeadPoseEstimation(**hpe_args)

        self.logger = logger
        self.visualize = args.visualize
        self.redo = args.redo_exist_result

    def __call__(self, paths: list) -> list:
        """Run face-area triming & head-pose-estimation

        Args:
            paths (list): [(video_path, hpe_result_path, triming_result_path, visualize_path1, visualize_path2), ...]

        Returns:
            list: [[HPE_area1.hp, HPE_area2.hp, ...], ...]
        """

        Loging_MSG.large_phase(self.logger, "START TRIMING PHASE")

        # triming area
        input_trim, _ = self.get_input_trim(paths)
        self.trimer(input_trim)

        Loging_MSG.large_phase(self.logger, "START HEAD-POSE-ESTIMATION PHASE")

        # head pose estimation
        input_hpe, _ = self.get_input_hpe(paths)
        hpe_result = self.hpe(input_hpe)

        # display hpe reslt files
        for result in hpe_result:
            self.logger.info(f"saved file {result}")

        return hpe_result

    def get_input_trim(self, paths: list) -> list:
        input_trim = []
        indx = []

        for i, path_set in enumerate(paths):

            if self.exist_result(path_set[2]) and not self.redo:
                continue

            output = None
            if self.visualize:
                if len(path_set) != 5:
                    ValueError(f"each phase args are expected 5, but got {len(paths)}")
                output = path_set[3]

            input_set = (path_set[0], output, path_set[2])
            input_trim.append(input_set)
            indx.append(i)

        return input_trim, indx

    def get_input_hpe(self, paths: list) -> list:
        input_hpe = []
        indx = []

        for i, path_set in enumerate(paths):

            if self.exist_result(path_set[1]) and not self.redo:
                continue

            area_set = load_face_area(path_set[2])

            output = None
            if self.visualize:
                if len(path_set) != 5:
                    ValueError(f"each phase args are expected 5, but got {len(paths)}")
                output = path_set[4]

            input_set = (path_set[0], path_set[1], area_set, output)
            input_hpe.append(input_set)
            indx.append(i)

        return input_hpe, indx

    def exist_result(self, path):
        """Check if the results of trim-area and face-mesh exist."""
        dir_path = os.path.dirname(path)
        target_file = os.path.basename(path)

        target_name, target_ext = target_file.split(".")
        target_name = target_name.split("_")

        file_list = os.listdir(dir_path)
        exist_flg = False
        for file in file_list:
            if not os.path.isfile(os.path.join(dir_path, file)):
                continue

            f_name, f_ext = file.split(".")
            f_name = f_name.split("_")

            if f_ext != target_ext:
                continue

            flg = False
            for idx, t_n in enumerate(target_name):
                if t_n != f_name[idx]:
                    flg = True
                    break
            if not flg:
                exist_flg = True
                break

        return exist_flg

    def get_arg_trim(self, logger: Logger, args: Namespace) -> dict:
        trim_args = {}
        trim_args["logger"] = logger
        trim_args["min_detection_confidence"] = args.min_detection_confidence
        trim_args["model_selection"] = args.model_selection
        trim_args["frame_step"] = args.frame_step
        trim_args["box_ratio"] = args.box_ratio
        trim_args["track_volatility"] = args.track_volatility
        trim_args["lost_volatility"] = args.lost_volatility
        trim_args["size_volatility"] = args.size_volatility
        trim_args["sub_track_volatility"] = args.sub_track_volatility
        trim_args["sub_size_volatility"] = args.sub_size_volatility
        trim_args["threshold"] = args.threshold
        trim_args["threshold_size_rate"] = args.threshold_size_rate
        trim_args["overlap"] = args.overlap
        trim_args["integrate_step"] = args.integrate_step
        trim_args["integrate_volatility"] = args.integrate_volatility
        trim_args["use_tracking"] = args.use_tracking
        trim_args["prohibit_integrate"] = args.prohibit_integrate
        trim_args["size_limit_rate"] = args.size_limit_rate
        trim_args["gc"] = args.gc
        trim_args["gc_term"] = args.gc_term
        trim_args["gc_success"] = args.gc_success
        trim_args["lost_track"] = args.lost_track
        trim_args["process_num"] = args.process_num
        trim_args["redo"] = args.redo_exist_result
        trim_args["visualize"] = args.visualize
        trim_args["single_process"] = args.sigle_process

        return trim_args

    def get_hpe_args(self, logger: Logger, args: Namespace) -> dict:
        hpe_args = {}

        hpe_args["logger"] = logger
        hpe_args["min_detection_confidence"] = args.min_detection_confidence
        hpe_args["min_tracking_confidence"] = args.min_tracking_confidence
        hpe_args["max_num_face"] = args.max_num_face
        hpe_args["redo"] = args.redo_exist_result
        hpe_args["visualize"] = args.visualize
        hpe_args["result_length"] = args.result_length
        hpe_args["batch_size"] = args.process_num

        return hpe_args
