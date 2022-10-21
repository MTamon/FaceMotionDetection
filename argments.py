from argparse import ArgumentParser


def add_args(parser: ArgumentParser):

    parser.add_argument(
        "--min_detection-confidence",
        default=0.7,
        type=float,
        help="Minimum confidence value ([0.0, 1.0]) for face detection to be considered successful. See details in https://solutions.mediapipe.dev/face_detection#min_detection_confidence.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        default=0.5,
        type=float,
        help="face landmarks to be considered tracked successfully. See details in https://solutions.mediapipe.dev/face_mesh#min_tracking_confidence.",
    )
    parser.add_argument(
        "--max-num-face",
        default=1,
        type=int,
        help="Maximum number of faces to detect. See details in https://solutions.mediapipe.dev/face_mesh#max_num_faces.",
    )
    parser.add_argument(
        "--model-selection",
        default=1,
        type=int,
        help="0 or 1. 0 to select a short-range model that works best for faces within 2 meters from the camera, and 1 for a full-range model best for faces within 5 meters. See details in https://solutions.mediapipe.dev/face_detection#model_selection.",
    )
    parser.add_argument(
        "--frame-step",
        default=1,
        type=int,
        help="Frame loading frequency. Defaults to 1.",
    )
    parser.add_argument(
        "--box-ratio",
        default=1.1,
        type=float,
        help="Face-detection bounding-box's rate. Defaults to 1.1.",
    )
    parser.add_argument(
        "--track-volatility",
        default=0.3,
        type=float,
        help="Volatility of displacement during face tracking. Defaults to 0.3.",
    )
    parser.add_argument(
        "--lost-volatility",
        default=0.1,
        type=float,
        help="Volatility of displacement when lost face tracking. Defaults to 0.1.",
    )
    parser.add_argument(
        "--size-volatility",
        default=0.03,
        type=float,
        help="Volatility in face size when face detection. Defaults to 0.03.",
    )
    parser.add_argument(
        "--sub-track-volatility",
        default=1.0,
        type=float,
        help="Volatility of the tracking decision when referring to the last detection position in non-tracking mode, regardless of the period of time lost. Defaults to 1.0.",
    )
    parser.add_argument(
        "--sub-size-volatility",
        default=0.5,
        type=float,
        help="Volatility of the tracking decision when referring to the last detection size in non-tracking mode, regardless of the period of time lost. Defaults to 0.5.",
    )
    parser.add_argument(
        "--threshold",
        default=0.3,
        type=float,
        help="Exclude clippings with low detection rates. Defaults to 0.3.",
    )
    parser.add_argument(
        "--overlap",
        default=0.9,
        type=float,
        help="Integration conditions in duplicate clippings. Defaults to 0.9.",
    )
    parser.add_argument(
        "--integrate-step",
        default=1,
        type=int,
        help="Integration running frequency. Defaults to 1.",
    )
    parser.add_argument(
        "--integrate-volatility",
        default=0.4,
        type=float,
        help="Allowable volatility of size features between two clippings when integrating clippings. Defaults to 0.4.",
    )
    parser.add_argument(
        "--use-tracking",
        default=False,
        action="store_true",
        help="Whether or not to use the face tracking feature. Defaults to True.",
    )
    parser.add_argument(
        "--prohibit-integrate",
        default=0.7,
        type=float,
        help="Threshold to prohibit integration of clippings. Defaults to 0.7.",
    )
    parser.add_argument(
        "--size-limit-rate",
        default=4,
        type=int,
        help="Maximum size of the clipping relative to the size of the detected face. Defaults to 4.",
    )
    parser.add_argument(
        "--gc",
        default=0.03,
        type=float,
        help="Success rate thresholds in garbage collect during gc_term. Defaults to 0.03.",
    )
    parser.add_argument(
        "--gc-term",
        default=100,
        type=int,
        help="Garbage collection execution cycle. Defaults to 100.",
    )
    parser.add_argument(
        "--gc-success",
        default=0.1,
        type=float,
        help="Success rate thresholds in garbage collect. Defaults to 0.1.",
    )
    parser.add_argument(
        "--lost-track",
        default=2,
        type=int,
        help="Number of turns of steps to search the vicinity when face detection is lost. Defaults to 2.",
    )
    parser.add_argument(
        "--process-num",
        default=3,
        type=int,
        help="Maximum number of processes in parallel processing. Defaults to 3.",
    )
    parser.add_argument(
        "--visualize",
        default=False,
        action="store_true",
        help="Visualization options for the analysis process. When use this option, processing speed is made be lower. Defaults to False.",
    )
    parser.add_argument(
        "--result-length",
        default=100000,
        type=int,
        help="result's max step number. memory saving effect. Defaults to 100000.",
    )

    return parser


def get_args():
    parser = ArgumentParser(
        "This program is experimental code for getting ability to impliment Face Mesh."
    )
    parser = add_args(parser)

    return parser.parse_args()
