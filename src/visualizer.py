from logging import Logger
from typing import Iterable, List
import cv2
import wave
import os
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
import moviepy.editor as mpedit

from src.utils import Video, CalcTools as tools


drawSpec = DrawingSpec(thickness=1, circle_radius=1, color=(244, 244, 244))

CUBE_SIZE = 600
CUBE = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
)
CUBE_CONTOURS = np.array(
    [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
)
BOTTOM_CONTOURS = np.array([4, 5, 7, 6])

FACE_OVAL = np.array(
    [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ]
)


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

        video.close_writer()

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
    def visualize_grads(results, result_title, stand_size, stand_rotate, hline_pos):
        def get_plot(key: str):
            x = []
            y = []
            for result in results:
                if result[key] is not None:
                    x.append(result["step"])
                    y.append(np.linalg.norm(result[key]))
            return {"x": x, "y": y}

        pg2 = get_plot("grad2")
        rg2 = get_plot("gradR2")
        zg1 = get_plot("gradZ1")

        fig = plt.figure(figsize=(20, 12.0))

        ax_pg2 = fig.add_subplot(311)
        ax_rg2 = fig.add_subplot(312)
        ax_zg1 = fig.add_subplot(313)

        plt.subplots_adjust(wspace=0.2, hspace=0.5)

        ax_pg2.set_title("position_grad2")
        ax_rg2.set_title("rotation_grad2")
        ax_zg1.set_title("size_grad1")

        ax_pg2.set_ylim(0.0, 0.2)
        ax_rg2.set_ylim(0.0, 15.0)
        ax_zg1.set_ylim(0.0, 0.1)

        ax_pg2.scatter(pg2["x"], pg2["y"], 1)
        ax_rg2.scatter(rg2["x"], rg2["y"], 1)
        ax_zg1.scatter(zg1["x"], zg1["y"], 1)

        ax_pg2.axhline(hline_pos, color="red")
        ax_rg2.axhline(stand_rotate, color="red")
        ax_zg1.axhline(stand_size, color="red")

        plt.savefig(result_title)
        plt.close()

    @staticmethod
    def interpolation(
        org_t: ndarray,
        org_x: ndarray,
        org_state: Iterable[str],
        mod_t: ndarray,
        mod_x: ndarray,
        mod_state: Iterable[str],
        curvs: ndarray,
        graph_time: ndarray,
        up_lim: float,
        low_lim: float,
        pict_path: str,
    ):
        if org_x.ndim == 1 and mod_x.ndim == 1:
            org_x = org_x.reshape((1, -1))
            mod_x = mod_x.reshape((1, -1))

        fig = plt.figure()
        disp_form = len(org_x) * 100 + 10
        plt.rcParams["figure.figsize"] = [20, 12.0]
        plt.subplots_adjust(wspace=0.2, hspace=0.5)

        for ax in range(len(org_x)):
            sub_plot = fig.add_subplot(disp_form + (ax + 1))
            sub_plot.set_title(f"ax-{ax} [{low_lim[ax]}, {up_lim[ax]}]")

            sub_plot.plot(graph_time, curvs[ax])

            normal_set = []
            masked_set = []
            noised_set = []
            for i, state in enumerate(org_state):
                org_set = [org_t[i], org_x[ax][i]]
                if state == "normal":
                    normal_set.append(org_set)
                if state == "masked":
                    masked_set.append(org_set)
                if state == "noised":
                    noised_set.append(org_set)
            normal_set = np.array(normal_set)
            masked_set = np.array(masked_set)
            noised_set = np.array(noised_set)

            if len(normal_set) != 0:
                sub_plot.scatter(
                    normal_set[:, 0], normal_set[:, 1], c="b", marker="+", s=100
                )
            if len(masked_set) != 0:
                sub_plot.scatter(
                    masked_set[:, 0], masked_set[:, 1], c="r", marker="+", s=100
                )
            if len(noised_set) != 0:
                sub_plot.scatter(
                    noised_set[:, 0], noised_set[:, 1], c="m", marker="+", s=100
                )

            normal_set = []
            masked_set = []
            noised_set = []
            for i, state in enumerate(mod_state):
                mod_set = [mod_t[i], mod_x[ax][i]]
                if state == "normal":
                    normal_set.append(mod_set)
                if state == "masked":
                    masked_set.append(mod_set)
                if state == "noised":
                    noised_set.append(mod_set)
            normal_set = np.array(normal_set)
            masked_set = np.array(masked_set)
            noised_set = np.array(noised_set)

            if len(normal_set) != 0:
                sub_plot.scatter(normal_set[:, 0], normal_set[:, 1], c="b", marker="o")
            if len(masked_set) != 0:
                sub_plot.scatter(masked_set[:, 0], masked_set[:, 1], c="r", marker="o")
            if len(noised_set) != 0:
                sub_plot.scatter(noised_set[:, 0], noised_set[:, 1], c="m", marker="o")

        plt.savefig(pict_path)
        plt.close()

    @staticmethod
    def _draw_cude(frame, norm_info, basic_dist):
        _frame = frame.copy()

        _CUBE = CUBE * CUBE_SIZE
        _CUBE = _CUBE - np.mean(_CUBE)

        _R = tools.rotation_matrix(*norm_info[1])
        centroid = norm_info[0]
        z = centroid[2]

        _z = z + basic_dist
        dist_volatility = _z / basic_dist
        ratio = 1 / dist_volatility**2

        _CUBE *= ratio

        _CUBE = np.dot(_R.T, _CUBE.T).T
        _CUBE += centroid

        bottom_points = []
        for point in BOTTOM_CONTOURS:
            bottom_points.append(_CUBE[point, :2])
        bottom_points = np.stack(bottom_points).astype(np.int32)
        bottom_points = bottom_points.reshape((1, -1, 2))
        cv2.fillPoly(frame, bottom_points, (50, 50, 50))
        for contours in CUBE_CONTOURS:
            strt = _CUBE[contours[0], :2].astype(np.int32)
            stop = _CUBE[contours[1], :2].astype(np.int32)

            cv2.line(frame, strt, stop, (255, 50, 255), 2)

        frame = cv2.addWeighted(frame, 0.4, _frame, 0.6, 0)

        return frame

    @staticmethod
    def shape_result(
        video: Video,
        result: ndarray,
        norm_info: ndarray,
        basic_dist: float,
        normalizer: float,
        path: str,
        tqdm_visual: bool,
    ):
        video.set_out_path(path)

        _R = tools.rotation_matrix(*norm_info[1])

        if tqdm_visual:
            progress_iterator = tqdm(result, desc="  visualize-all ")
        else:
            progress_iterator = result

        for step_result, frame in zip(progress_iterator, video):
            frame = Visualizer._draw_cude(frame, norm_info, basic_dist)

            if step_result["ignore"]:
                Visualizer.frame_writer(frame, video)
                continue

            head_direction = np.array([0.0, 0.0, -1.0])

            centroid = step_result["centroid"] / normalizer
            R = step_result["rotate"]
            landmarks = step_result["countenance"]
            ratio = step_result["ratio"]

            # Release normalization
            R = np.dot(_R, R)
            centroid = np.dot(_R.T, centroid) + norm_info[0]

            clr = (255, 255, 255)
            if step_result["masked"]:
                clr = (255, 255, 100)

            landmarks /= ratio
            landmarks = np.dot(R.T, landmarks.T).T + centroid
            angle = tools.rotation_angles(R)

            frame = Visualizer._draw_cude(frame, (centroid, angle), basic_dist)

            for pt in landmarks:
                cv2.drawMarker(frame, (int(pt[0]), int(pt[1])), clr, cv2.MARKER_STAR, 2)

            cv2.putText(
                img=frame,
                text=f"depth: {round(centroid[2], 2)}",
                org=(20, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5,
                color=(0, 255, 0),
                thickness=2,
            )
            x, y, z = angle
            cv2.putText(
                frame,
                "x: " + str(np.round(x, 2)),
                (500, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "y: " + str(np.round(y, 2)),
                (500, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "z: " + str(np.round(z, 2)),
                (500, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            head_direction = np.dot(R.T, head_direction)
            head_direction *= 200
            nose = (int(landmarks[1][0]), int(landmarks[1][1]))
            end = (nose[0] + int(head_direction[0]), nose[1] + int(head_direction[1]))
            cv2.line(frame, nose, end, (255, 50, 50), 3)

            Visualizer.frame_writer(frame, video)

        video.close_writer()

    @staticmethod
    def shape_removed(
        video: Video,
        results: np.ndarray,
        hme_result: np.ndarray,
        threshold_size: float,
        threshold_rotate: float,
        threshold_pos: float,
        path: str,
        tqdm_visual: bool = False,
    ):

        video.set_out_path(path)

        if tqdm_visual:
            progress_iterator = enumerate(tqdm(video, desc="      visualize "))
        else:
            progress_iterator = enumerate(video)

        for i, frame in progress_iterator:
            step_hme_result = hme_result[i]
            step_result = results[i]
            face_mesh = step_hme_result["landmarks"]

            if face_mesh is not None:

                for landmark in face_mesh:
                    landmark = landmark.astype(np.int32)

                    clr = (255, 255, 255)
                    flg_none = True

                    if step_result["gradZ1"] is not None:
                        flg_none = False
                        if step_result["gradZ1"] > threshold_size:
                            clr = (50, 50, 255)
                    if step_result["grad2"] is not None:
                        flg_none = False
                        if np.linalg.norm(step_result["grad2"]) > threshold_pos:
                            clr = (0, 0, 0)
                    if step_result["gradR2"] is not None:
                        flg_none = False
                        if step_result["gradR2"] > threshold_rotate:
                            clr = (50, 255, 50)
                    if flg_none:
                        clr = (255, 50, 50)

                    cv2.drawMarker(
                        frame, (landmark[0], landmark[1]), clr, cv2.MARKER_STAR, 2
                    )
            Visualizer.frame_writer(frame, video)

        video.close_writer()

    @staticmethod
    def audio_visual_matching(logger: Logger, match_info: dict):
        for record in match_info["pairs"]:
            sh_path = record["sh"]
            wav_path = record["wav"]
            mp4_ALL_path = sh_path[:-3] + "ALL.mp4"
            if not os.path.isfile(mp4_ALL_path):
                logger.warn(f"No such a mp4 file: {mp4_ALL_path}")
                continue

            wave_file = wave.open(wav_path, "r")
            fps = wave_file.getframerate()
            nbytes = wave_file.getsampwidth()

            mp4_MTH_path = mp4_ALL_path[:-7] + "MTH.mp4"
            temp_dir, temp_file = os.path.split(mp4_MTH_path[:-4] + "TEMP_.mp3")
            temp_dir = temp_dir + "/temp"
            temp_path = temp_dir + "/" + temp_file
            if not os.path.isdir(temp_dir):
                os.mkdir(temp_dir)

            video = mpedit.VideoFileClip(mp4_ALL_path)
            video = video.set_audio(
                mpedit.AudioFileClip(wav_path, fps=fps, nbytes=nbytes)
            )

            video.write_videofile(mp4_MTH_path, temp_audiofile=temp_path)
