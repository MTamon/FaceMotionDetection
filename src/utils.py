from logging import Logger
import math
import os
import cv2
import numpy as np

from typing import Iterable, Tuple


class Video:
    def __init__(self, video_path: str, codec: str = "mp4v") -> None:
        self.cap = cv2.VideoCapture(video_path)
        self.fourcc = cv2.VideoWriter_fourcc(*codec)

        self.path = video_path
        self.name = os.path.basename(video_path)
        self.codec = codec

        self.cap_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cap_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.cap_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.writer = None

        self.step = 1

        self.current_idx = 0

        self.length = None
        self.__len__()

    def __str__(self) -> str:
        return f"all frame : {self.cap_frames}, fps : {round(self.fps, 2)}, time : {round(self.cap_frames/self.fps, 2)}"

    def __getitem__(self, idx):
        pos = cv2.CAP_PROP_POS_FRAMES
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        return ret, frame

    def __len__(self) -> int:
        if self.length is None:
            self.length = math.ceil(self.cap_frames / self.step)
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx == self.length:
            raise StopIteration

        frame = self.cap.read()[1]
        self.current_idx += 1
        for _ in range(self.step - 1):
            self.cap.read()
        return frame

    def reset(self):
        self.current_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def info(self):
        return [self.fourcc, self.cap_width, self.cap_height, self.fps, self.cap_frames]

    def read(self):
        return self.cap.read()

    def set_out_path(self, path: str):
        self.writer = cv2.VideoWriter(
            path, self.fourcc, self.fps, (self.cap_width, self.cap_height)
        )

    def write(self, frame):
        self.writer.write(frame)

    def set_step(self, step):
        self.step = step

    def close_writer(self):
        self.writer.release()


class Loging_MSG:
    @staticmethod
    def large_phase(logger: Logger, msg: str):
        len_str = len(msg)
        len_wall = 80

        if len_str > 70:
            len_wall = len_str + 10

        l_spaces = " " * int((70 - len_str) / 2)
        r_spaces = " " * int((70 - len_str) / 2 + 0.5)

        logger.info("#" * len_wall)
        logger.info(f"##   {l_spaces}{msg}{r_spaces}   ##")
        logger.info("#" * len_wall)


class CalcTools:
    """For calculation tools class"""

    @staticmethod
    def local2global(
        area: dict, frame_size: Iterable[int], coordinate: Iterable[float]
    ) -> Tuple[float]:
        """Transform facemesh coordinate, area-coordinate -> frame-coordinate"""
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

    @staticmethod
    def matrix_to_angles(matrixes: np.ndarray) -> np.ndarray:
        """This 'matrixes' is a rotation matrix
        that rotates the face from the front to the actual direction
        """
        angles = []
        # This R is a rotation matrix that rotates the face from the front to the actual direction
        for R in matrixes:
            x, y, z = CalcTools.rotation_angles(R)
            angles.append([x, y, z])  # 360 degree base

        return np.array(angles)

    @staticmethod
    def angles_to_matrix(angles: np.ndarray) -> np.ndarray:
        """This 'angles' is a euler angle
        that rotates the face from the front to the actual direction
        """
        matrixes = []
        # This angle is a rotation matrix that rotates the face from the front to the actual direction
        for angle in angles:
            R = CalcTools.rotation_matrix(*angle)
            matrixes.append(R)

        return np.stack(matrixes)

    @staticmethod
    def rotation_matrix(
        theta1: float, theta2: float, theta3: float, order="xyz"
    ) -> np.ndarray:
        """
        入力
            theta1, theta2, theta3 = Angle of rotation theta 1, 2, 3 in order of rotation
            oreder = Order of rotation e.g. 'xzy' for X, Z, Y order
        出力
            3x3 Rotation Matrix
        """
        c1 = np.cos(theta1 * np.pi / 180)
        s1 = np.sin(theta1 * np.pi / 180)
        c2 = np.cos(theta2 * np.pi / 180)
        s2 = np.sin(theta2 * np.pi / 180)
        c3 = np.cos(theta3 * np.pi / 180)
        s3 = np.sin(theta3 * np.pi / 180)

        if order == "xzx":
            matrix = np.array(
                [
                    [c2, -c3 * s2, s2 * s3],
                    [c1 * s2, c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3],
                    [s1 * s2, c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3],
                ]
            )
        elif order == "xyx":
            matrix = np.array(
                [
                    [c2, s2 * s3, c3 * s2],
                    [s1 * s2, c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1],
                    [-c1 * s2, c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3],
                ]
            )
        elif order == "yxy":
            matrix = np.array(
                [
                    [c1 * c3 - c2 * s1 * s3, s1 * s2, c1 * s3 + c2 * c3 * s1],
                    [s2 * s3, c2, -c3 * s2],
                    [-c3 * s1 - c1 * c2 * s3, c1 * s2, c1 * c2 * c3 - s1 * s3],
                ]
            )
        elif order == "yzy":
            matrix = np.array(
                [
                    [c1 * c2 * c3 - s1 * s3, -c1 * s2, c3 * s1 + c1 * c2 * s3],
                    [c3 * s2, c2, s2 * s3],
                    [-c1 * s3 - c2 * c3 * s1, s1 * s2, c1 * c3 - c2 * s1 * s3],
                ]
            )
        elif order == "zyz":
            matrix = np.array(
                [
                    [c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3, c1 * s2],
                    [c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3, s1 * s2],
                    [-c3 * s2, s2 * s3, c2],
                ]
            )
        elif order == "zxz":
            matrix = np.array(
                [
                    [c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1, s1 * s2],
                    [c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3, -c1 * s2],
                    [s2 * s3, c3 * s2, c2],
                ]
            )
        elif order == "xyz":
            matrix = np.array(
                [
                    [c2 * c3, -c2 * s3, s2],
                    [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
                    [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2],
                ]
            )
        elif order == "xzy":
            matrix = np.array(
                [
                    [c2 * c3, -s2, c2 * s3],
                    [s1 * s3 + c1 * c3 * s2, c1 * c2, c1 * s2 * s3 - c3 * s1],
                    [c3 * s1 * s2 - c1 * s3, c2 * s1, c1 * c3 + s1 * s2 * s3],
                ]
            )
        elif order == "yxz":
            matrix = np.array(
                [
                    [c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3, c2 * s1],
                    [c2 * s3, c2 * c3, -s2],
                    [c1 * s2 * s3 - c3 * s1, c1 * c3 * s2 + s1 * s3, c1 * c2],
                ]
            )
        elif order == "yzx":
            matrix = np.array(
                [
                    [c1 * c2, s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3],
                    [s2, c2 * c3, -c2 * s3],
                    [-c2 * s1, c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3],
                ]
            )
        elif order == "zyx":
            matrix = np.array(
                [
                    [c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
                    [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
                    [-s2, c2 * s3, c2 * c3],
                ]
            )
        elif order == "zxy":
            matrix = np.array(
                [
                    [c1 * c3 - s1 * s2 * s3, -c2 * s1, c1 * s3 + c3 * s1 * s2],
                    [c3 * s1 + c1 * s2 * s3, c1 * c2, s1 * s3 - c1 * c3 * s2],
                    [-c2 * s3, s2, c2 * c3],
                ]
            )

        return matrix

    @staticmethod
    def rotation_angles(matrix: np.ndarray, order: str = "xyz") -> Tuple[float]:
        """
        Parameters
            matrix = 3x3 Rotation Matrix
            oreder = Order of rotation e.g. 'xzy' for X, Z, Y order
        Outputs
            theta1, theta2, theta3 = Angle of rotation theta 1, 2, 3 in order of rotation
        """
        r11, r12, r13 = matrix[0]
        r21, r22, r23 = matrix[1]
        r31, r32, r33 = matrix[2]

        if order == "xzx":
            theta1 = np.arctan(r31 / r21)
            theta2 = np.arctan(r21 / (r11 * np.cos(theta1)))
            theta3 = np.arctan(-r13 / r12)

        elif order == "xyx":
            theta1 = np.arctan(-r21 / r31)
            theta2 = np.arctan(-r31 / (r11 * np.cos(theta1)))
            theta3 = np.arctan(r12 / r13)

        elif order == "yxy":
            theta1 = np.arctan(r12 / r32)
            theta2 = np.arctan(r32 / (r22 * np.cos(theta1)))
            theta3 = np.arctan(-r21 / r23)

        elif order == "yzy":
            theta1 = np.arctan(-r32 / r12)
            theta2 = np.arctan(-r12 / (r22 * np.cos(theta1)))
            theta3 = np.arctan(r23 / r21)

        elif order == "zyz":
            theta1 = np.arctan(r23 / r13)
            theta2 = np.arctan(r13 / (r33 * np.cos(theta1)))
            theta3 = np.arctan(-r32 / r31)

        elif order == "zxz":
            theta1 = np.arctan(-r13 / r23)
            theta2 = np.arctan(-r23 / (r33 * np.cos(theta1)))
            theta3 = np.arctan(r31 / r32)

        elif order == "xzy":
            theta1 = np.arctan(r32 / r22)
            theta2 = np.arctan(-r12 * np.cos(theta1) / r22)
            theta3 = np.arctan(r13 / r11)

        elif order == "xyz":
            theta1 = np.arctan(-r23 / r33)
            theta2 = np.arctan(r13 * np.cos(theta1) / r33)
            theta3 = np.arctan(-r12 / r11)

        elif order == "yxz":
            theta1 = np.arctan(r13 / r33)
            theta2 = np.arctan(-r23 * np.cos(theta1) / r33)
            theta3 = np.arctan(r21 / r22)

        elif order == "yzx":
            theta1 = np.arctan(-r31 / r11)
            theta2 = np.arctan(r21 * np.cos(theta1) / r11)
            theta3 = np.arctan(-r23 / r22)

        elif order == "zyx":
            theta1 = np.arctan(r21 / r11)
            theta2 = np.arctan(-r31 * np.cos(theta1) / r11)
            theta3 = np.arctan(r32 / r33)

        elif order == "zxy":
            theta1 = np.arctan(-r12 / r22)
            theta2 = np.arctan(r32 * np.cos(theta1) / r22)
            theta3 = np.arctan(-r31 / r33)

        theta1 = theta1 * 180 / np.pi
        theta2 = theta2 * 180 / np.pi
        theta3 = theta3 * 180 / np.pi

        return (theta1, theta2, theta3)
