from typing import List

import cv2
from mediapipe.python.solutions.drawing_utils import (
    DrawingSpec,
    draw_detection,
    draw_landmarks,
)
from mediapipe.python.solutions.face_mesh import FACEMESH_CONTOURS
from numpy import ndarray
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from src.utils import Video


drawSpec = DrawingSpec(thickness=1, circle_radius=1, color=(244, 244, 244))


class Visualizer:
    @staticmethod
    def face_area(face_area: List[dict], video: Video):
        frame_width = video.cap_width
        frame_height = video.cap_height

        for frame in tqdm(video, desc=video.name.split(".")[0]):
            for area in face_area:
                frame.flags.writeable = True

                pxcel_x_min = int(area["xmin"] * frame_width)
                pxcel_y_min = int(area["ymin"] * frame_height)
                pxcel_x_max = int((area["xmin"] + area["width"]) * frame_width)
                pxcel_y_max = int((area["ymin"] + area["height"]) * frame_height)
                pt1 = (pxcel_x_min, pxcel_y_min)
                pt2 = (pxcel_x_max, pxcel_y_max)
                clr = (0, 0, 255) if area["comp"] else (255, 255, 0)
                cv2.rectangle(
                    frame, pt1, pt2, clr, thickness=1, lineType=cv2.LINE_8, shift=0
                )

            Visualizer.frame_writer(frame, video)

    @staticmethod
    def face_area_window(
        face_area: List[dict],
        video: Video,
        frame,
        results=None,
        compatible_ids: list = None,
    ):
        frame_width = video.cap_width
        frame_height = video.cap_height

        # draw face_area
        for area in face_area:
            frame.flags.writeable = True

            pxcel_x_min = int(area["xmin"] * frame_width)
            pxcel_y_min = int(area["ymin"] * frame_height)
            pxcel_x_max = int((area["xmin"] + area["width"]) * frame_width)
            pxcel_y_max = int((area["ymin"] + area["height"]) * frame_height)
            pt1 = (pxcel_x_min, pxcel_y_min)
            pt2 = (pxcel_x_max, pxcel_y_max)
            clr = (255, 255, 0) if area["prev"] is None else (0, 0, 255)
            cv2.rectangle(
                frame, pt1, pt2, clr, thickness=1, lineType=cv2.LINE_8, shift=0
            )

        # draw mediapipe's original face detection result
        if results.detections is not None:
            for result in results.detections:
                draw_detection(frame, result, drawSpec, drawSpec)

        # draw compatible id faces
        if compatible_ids is not None:
            for (_, _), face in compatible_ids:
                center = (
                    int(face["xmin"] * frame_width),
                    int(face["ymin"] * frame_height),
                )
                cv2.circle(
                    frame, center=center, radius=3, color=(244, 244, 244), thickness=1
                )

        Visualizer.frame_writer(frame, video)

    @staticmethod
    def head_pose_plotter(frame: ndarray, head_pose: dict):

        if not head_pose["activation"]:
            return frame

        frame_h, frame_w, _ = frame.shape

        area_info = head_pose["area"]
        origin = head_pose["origin"]
        R = head_pose["angles"]
        landmarks = head_pose["landmarks"]

        area_width = int(area_info["width"] * frame_w)
        area_height = int(area_info["height"] * frame_h)
        area_xmin = int(area_info["xmin"] * frame_w)
        area_ymin = int(area_info["ymin"] * frame_h)

        x_lw = area_xmin
        x_up = area_xmin + area_width
        y_lw = area_ymin
        y_up = area_ymin + area_height

        area_frame = frame[y_lw:y_up, x_lw:x_up]

        pxcel_x = int(origin[0] * area_width)
        pxcel_y = int(origin[1] * area_height)

        head_direction = np.array([0, 0, -1]) * 200
        head_direction = np.dot(R, head_direction)

        p1 = (pxcel_x, pxcel_y)
        p2 = (p1[0] + int(head_direction[0]), p1[1] + int(head_direction[1]))

        cv2.line(area_frame, p1, p2, color=(255, 0, 0), thickness=3)
        draw_landmarks(
            image=area_frame,
            landmark_list=landmarks,
            connections=FACEMESH_CONTOURS,
            landmark_drawing_spec=drawSpec,
            connection_drawing_spec=drawSpec,
        )

        frame[y_lw:y_up, x_lw:x_up] = area_frame

        cv2.putText(
            img=frame,
            text=f"depth: {origin[2]}",
            org=(20, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.5,
            color=(0, 255, 0),
            thickness=2,
        )

        return frame

    @staticmethod
    def frame_writer(frame: ndarray, param: Video):
        param.write(frame)

    @staticmethod
    def visualize_grads(results, result_title):
        pg1_x = []
        pg1_y = []
        pg2_x = []
        pg2_y = []
        rg1_x = []
        rg1_y = []
        rg2_x = []
        rg2_y = []

        for result in results:

            if result["grad1"] is not None:
                pg1_x.append(result["step"])
                pg1_y.append(np.linalg.norm(result["grad1"]))
            if result["grad2"] is not None:
                pg2_x.append(result["step"])
                pg2_y.append(np.linalg.norm(result["grad2"]))
            if result["gradR1"] is not None:
                rg1_x.append(result["step"])
                rg1_y.append(result["gradR1"])
            if result["gradR2"] is not None:
                rg2_x.append(result["step"])
                rg2_y.append(result["gradR2"])

        fig = plt.figure()

        ax_pg1 = fig.add_subplot(221)
        ax_pg2 = fig.add_subplot(222)
        ax_rg1 = fig.add_subplot(223)
        ax_rg2 = fig.add_subplot(224)

        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        ax_pg1.set_title("position_grad1")
        ax_pg2.set_title("position_grad2")
        ax_rg1.set_title("rotation_grad1")
        ax_rg2.set_title("rotation_grad2")

        ax_pg1.scatter(pg1_x, pg1_y, 2)
        ax_pg2.scatter(pg2_x, pg2_y, 2)
        ax_rg1.scatter(rg1_x, rg1_y, 2)
        ax_rg2.scatter(rg2_x, rg2_y, 2)

        plt.savefig(result_title)
