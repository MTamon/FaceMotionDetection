"""Extraction Phase for face motion data."""

from argparse import Namespace
from logging import Logger
from .trim.triming_area import TrimFace
from .face_mesh.face_mesh import HeadPoseEstimation
from .io import write_face_area


class Extraction:
    def __init__(self, logger: Logger, args: Namespace) -> None:
        trim_args = self.get_arg_trim(logger, args)
        hpe_args = self.get_hpe_args(logger, args)

        self.trimer = TrimFace(**trim_args)
        self.hpe = HeadPoseEstimation(**hpe_args)

        self.logger = logger
        self.visualize = args.visualize

    def __call__(self, paths: list) -> list:
        """Run face-area triming & head-pose-estimation

        Args:
            paths (list): [(video_path, hpe_result_path, triming_result_path, visualize_path), ...]

        Returns:
            list: [[HPE_area1.hp, HPE_area2.hp, ...], ...]
        """

        self.logger.info("#" * 80)
        self.logger.info("#  START TRIMING PHASE")
        self.logger.info("#" * 80)

        # triming area
        input_trim = self.get_input_trim(paths)
        trimer_result = self.trimer(input_trim)

        # save triming result
        _trimer_result = []
        for idx, (result, fpath) in enumerate(zip(trimer_result, paths)):
            _trimer_result.append(write_face_area(fpath[2], result))
        trimer_result = _trimer_result

        self.logger.info("#" * 80)
        self.logger.info("#  START HEAD-POSE-ESTIMATION PHASE")
        self.logger.info("#" * 80)

        # head pose estimation
        input_hpe = self.get_input_hpe(paths, trimer_result)
        hpe_result = self.hpe(input_hpe)

        # display hpe reslt files
        for results in hpe_result:
            for result in results:
                self.logger.info(f"saved file {result}")

        return hpe_result

    def get_input_trim(self, paths: list) -> list:
        input_trim = []

        for path_set in paths:

            output = None
            if self.visualize:
                if len(paths) < 4:
                    ValueError("visualize-mode needs visualize-path")
                output = path_set[3]

            input_set = (path_set[0], output)
            input_trim.append(input_set)

        return input_trim

    def get_input_hpe(self, paths: list, areas: list) -> list:
        input_hpe = []

        for path_set, area_set in zip(paths, areas):

            output = None
            if self.visualize:
                if len(paths) < 4:
                    ValueError("visualize-mode needs visualize-path")
                output = path_set[3]

            input_set = (path_set[0], path_set[1], area_set, output)
            input_hpe.append(input_set)

        return input_hpe

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
        trim_args["visualize"] = args.visualize

        return trim_args

    def get_hpe_args(self, logger: Logger, args: Namespace) -> dict:
        hpe_args = {}

        hpe_args["logger"] = logger
        hpe_args["min_detection_confidence"] = args.min_detection_confidence
        hpe_args["min_tracking_confidence"] = args.min_tracking_confidence
        hpe_args["max_num_face"] = args.max_num_face
        hpe_args["visualize"] = args.visualize
        hpe_args["override_input"] = False
        hpe_args["result_length"] = args.result_length

        return hpe_args
