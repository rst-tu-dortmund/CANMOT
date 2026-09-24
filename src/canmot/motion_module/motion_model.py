"""
motion model of trajectory, notice that objects of different categories often exhibit various motion patterns.
Paper motion models: Constant Velocity (CV), Constant Acceleration (CA), and
Constant Turn Rate and Acceleration (CTRA).
"""

import abc
import numpy as np
from pyquaternion import Quaternion
from canmot.data.script.NUSC_CONSTANT import *
from canmot.utils.math import warp_to_pi
class ABC_MODEL(abc.ABC):
    """interface of all motion models"""

    def __init__(self) -> None:
        self.SD = self.MD = -1
        self.angle_valued_state_idcs = []
        self.angle_valued_measure_idcs = []

    def stateTransition(self, state: np.ndarray) -> np.ndarray:
        """state transition function, non-linear model need to override this function

        Args:
            state (np.ndarray): [state dim, 1], state vector

        Returns:
            np.ndarray: [state dim, 1], state vector after transition
        """
        return self.getTransitionF(state) @ state

    def StateToMeasure(self, state: np.ndarray) -> np.ndarray:
        """state to measure function, non-linear model need to override this function

        Args:
            state (np.ndarray): [state dim, 1], state vector

        Returns:
            np.ndarray: [measure dim, 1], measure vector
        """
        return self.getMeaStateH(state) @ state

    @abc.abstractmethod
    def getInitState(self, det_infos: dict) -> np.ndarray:
        """from detection init tracklet

        Args:
            det_infos (dict): detection infos under different data format.
            {
                'nusc_box': NuscBox,
                'np_array': np.array,
                'has_velo': bool, whether the detetor has velocity info
            }

        Returns:
            np.ndarray: [state dim, 1], state vector
        """
        pass

    @abc.abstractmethod
    def getInitCovP(self, cls_label: int) -> np.ndarray:
        """init errorcov.

        Args:
            cls_label (int): set init errorcov category-specific.

        Returns:
            np.ndarray: [state dim, state dim], Initialized covariance matrix
        """
        pass

    @abc.abstractmethod
    def getProcessNoiseQ(self, cls_label: int) -> np.ndarray:
        """get process noise matrix. The value set is somewhat arbitrary

        Returns:
            np.ndarray: process noise matrix(fix)
        """
        pass

    @abc.abstractmethod
    def getTransitionF(self, state: np.ndarray) -> np.ndarray:
        """get state transition matrix.
        obtain matrix in the motion_module/script/
        Returns:
            np.ndarray: [state dim, state dim], state transition matrix
        """
        pass

    @abc.abstractmethod
    def getMeaNoiseR(self, cls_label: int) -> np.ndarray:
        """get measurement noise matrix. The value set is also somewhat arbitrary
        Returns:
            np.ndarray: measure noise matrix(fix)
        """
        pass

    @abc.abstractmethod
    def getMeaStateH(self, state: np.ndarray) -> np.ndarray:
        """get state to measure transition matrix.
        obtain matrix in the motion_module/script
        Returns:
            np.ndarray: [measure dim, state dim], state to measure transition matrix
        """
        pass

    @abc.abstractmethod
    def getOutputInfo(self, state: np.ndarray) -> np.ndarray:
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        Args:
            state (np.ndarray): [state dim, 1], predict or update state estimated by the filter

        Returns:
            np.array: [12(fix), 1], predict or update state under output file format
            output format: [x, y, z, w, l, h, vx, vy, ry(orientation, 1x4)]
        """
        pass

    def getStateDim(self) -> int:
        return self.SD

    def getMeasureDim(self) -> int:
        return self.MD

    def get_rotation_matrix_for_state(self, state: np.ndarray) -> np.ndarray:
        """get rotation matrix from state yaw angle

        Args:
            state (np.ndarray): [state dim, 1], state vector"""
        ry = state[-1, 0]
        cos_ry, sin_ry = np.cos(ry), np.sin(ry)
        R_xyz = np.array([[cos_ry, -sin_ry, 0], [sin_ry, cos_ry, 0], [0, 0, 1]])
        R = np.eye(self.SD)
        R[0:3, 0:3] = R_xyz
        return R

    def warpStateToPi(self, state: np.ndarray) -> np.ndarray:
        """warp angle-valued states to [-pi, pi]

        Args:
            state (np.ndarray): [state dim, 1], state vector

        Returns:
            np.ndarray: [state dim, 1], warped state vector
        """
        for idx in self.angle_valued_state_idcs:
            state[idx, 0] = warp_to_pi(state[idx, 0])
        return state

    def warpMeasureToPi(self, measure: np.ndarray) -> np.ndarray:
        """warp angle-valued measures to [-pi, pi]

        Args:
            measure (np.ndarray): [measure dim, 1], measure vector

        Returns:
            np.ndarray: [measure dim, 1], warped measure vector
        """
        angs = [warp_to_pi(measure[idx, 0]) for idx in self.angle_valued_measure_idcs]
        measure[self.angle_valued_measure_idcs, 0] = angs
        for idx in self.angle_valued_measure_idcs:
            measure[idx, 0] = warp_to_pi(measure[idx, 0])
        return measure


class CA(ABC_MODEL):
    """Constant Acceleration Motion Model
    Basic info:
        State vector: [x, y, z, w, l, h, vx, vy, vz, ax, ay, az, ry]
        Measure vector: [x, y, z, w, l, h, (vx, vy, optional), ry]
    """

    def __init__(self, has_velo: bool, dt: float, matrix_q=None, matrix_r=None) -> None:
        super().__init__()
        self.has_velo, self.dt, self.SD = has_velo, dt, 13
        self.MD = 9 if self.has_velo else 7

    def getInitState(self, det_infos: dict) -> np.ndarray:
        """from detection init tracklet
        Acceleration and velocity on the z-axis are both set to 0
        """
        init_state = np.zeros(shape=self.SD)
        det, det_box = det_infos["np_array"], det_infos["nusc_box"]

        # set x, y, z, w, l, h, (vx, vy, if velo is valid)
        init_state[:6] = det[:6]
        if self.has_velo:
            init_state[6:8] = det[6:8]

        # set yaw
        init_state[-1] = det_box.yaw

        # only for debug
        q = Quaternion(det[8:12].tolist())
        q = -q if q.axis[-1] < 0 else q
        assert q.radians == det_box.yaw

        return init_state[:, None]

    def getInitCovP(self, cls_label: int) -> np.ndarray:
        """init errorcov. Generally, the CA model can converge quickly,
        so not particularly sensitive to initialization
        """

        cls_name = CLASS_STR_TO_SEG_CLASS[cls_label]
        vector_p = (
            CA_INIT_EKFP[cls_name]
            if cls_name in CA_INIT_EKFP
            else CA_INIT_EKFP["default"]
        )

        return np.diag(vector_p)

    def getProcessNoiseQ(self, cls_label: int) -> np.ndarray:
        cls_name = CLASS_STR_TO_SEG_CLASS[cls_label]

        vector_q = (
            CA_INIT_EKFQ[cls_name]
            if cls_name in CA_INIT_EKFQ
            else CA_INIT_EKFQ["default"]
        )

        return np.diag(vector_q)

    def getMeaNoiseR(self, cls_label: int) -> np.ndarray:
        cls_name = CLASS_STR_TO_SEG_CLASS[cls_label]

        vector_r = (
            CA_INIT_EKFR[cls_name]
            if cls_name in CA_INIT_EKFR
            else CA_INIT_EKFR["default"]
        )

        return np.diag(vector_r)

    def getTransitionF(self, state=None) -> np.ndarray:
        """obtain matrix in the motion_module/script/Linear_kinect_jacobian.ipynb"""
        dt = self.dt
        F = np.asarray(
            [
                [1, 0, 0, 0, 0, 0, dt, 0, 0, 0.5 * dt**2, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0.5 * dt**2, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        return F

    def getMeaStateH(self, state=None) -> np.ndarray:
        """obtain matrix in the motion_module/script/Linear_kinect_jacobian.ipynb"""
        if self.has_velo:
            H = np.asarray(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )
        else:
            H = np.asarray(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )
        return H

    def getOutputInfo(self, state: np.ndarray) -> np.array:
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        """
        rotation = Quaternion(axis=(0, 0, 1), radians=state[-1, 0]).q
        list_state = state.T.tolist()[0][:8] + rotation.tolist()
        return np.array(list_state)

    @staticmethod
    def warpResYawToPi(res: np.ndarray) -> np.ndarray:
        """warp res yaw to [-pi, pi) in place

        Args:
            res (np.ndarray): [measure dim, 1]
            res infos -> [x, y, z, w, l, h, (vx, vy, optional), ry]

        Returns:
            np.ndarray: [measure dim, 1], residual warped to [-pi, pi)
        """
        res[-1, 0] = warp_to_pi(res[-1, 0])
        return res

    @staticmethod
    def warpStateYawToPi(state: np.ndarray) -> np.ndarray:
        """warp state yaw to [-pi, pi) in place

        Args:
            state (np.ndarray): [state dim, 1]
            State vector: [x, y, z, w, l, h, vx, vy, vz, ax, ay, az, ry]

        Returns:
            np.ndarray: [state dim, 1], state after warping
        """
        state[-1, 0] = warp_to_pi(state[-1, 0])
        return state


class CV(ABC_MODEL):
    """Constant Velocity Motion Model
    Basic info:
        State vector: [x, y, z, w, l, h, vx, vy, vz, ry]
        Measure vector: [x, y, z, w, l, h, (vx, vy, optional), ry]
    """

    def __init__(self, has_velo: bool, dt: float, matrix_q=None, matrix_r=None) -> None:
        super().__init__()
        self.has_velo, self.dt, self.SD = has_velo, dt, 10
        self.MD = 9 if self.has_velo else 7
        self.matrix_q = matrix_q
        self.matrix_r = matrix_r

    def get_rotation_matrix_for_state(self, state):
        """returns a matrix that rotates the first two components of the state and leaves the rest unchanged"""
        yaw = state[-1, 0]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        R = np.eye(self.SD)
        R[0, 0] = cos_yaw
        R[0, 1] = -sin_yaw
        R[1, 0] = sin_yaw
        R[1, 1] = cos_yaw
        R[6, 6] = cos_yaw
        R[6, 7] = -sin_yaw
        R[7, 6] = sin_yaw
        R[7, 7] = cos_yaw
        return R

    def get_rotation_matrix_for_measurement(self, state):
        """returns a matrix that rotates position and velocity components of the measurement according to the yaw angle in the state, and leaves the rest unchanged"""
        yaw = state[-1, 0]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        R = np.eye(self.MD)
        R[0, 0] = cos_yaw
        R[0, 1] = -sin_yaw
        R[1, 0] = sin_yaw
        R[1, 1] = cos_yaw

        if self.has_velo:
            R[6, 6] = cos_yaw
            R[6, 7] = -sin_yaw
            R[7, 6] = sin_yaw
            R[7, 7] = cos_yaw

        return R

    def getInitState(self, det_infos: dict) -> np.ndarray:
        """from detection init tracklet
        Velocity on the z-axis are set to 0
        """
        init_state = np.zeros(shape=self.SD)
        det, det_box = det_infos["np_array"], det_infos["nusc_box"]

        # set x, y, z, w, l, h, (vx, vy, if velo is valid)
        init_state[:6] = det[:6]
        if self.has_velo:
            init_state[6:8] = det[6:8]

        # set yaw
        init_state[-1] = det_box.yaw

        # only for debug
        q = Quaternion(det[8:12].tolist())
        q = -q if q.axis[-1] < 0 else q
        assert q.radians == det_box.yaw

        return np.asarray(init_state)[:, None]

    def getInitCovP(self, cls_label: int) -> np.ndarray:
        """init errorcov. Generally, the CV model can converge quickly,
        so not particularly sensitive to initialization
        """
        if self.matrix_r is None:
            return np.asarray(np.eye(self.SD)) * 0.01
        # H is state-independent for the linear CV model, so a placeholder state suffices
        H = self.getMeaStateH(np.zeros((self.SD, 1)))
        P = H.T @ np.asarray(self.matrix_r[cls_label]) @ H
        P[8, 8] = 0.1  # set the vertical velocity variance to a small value
        return P

    def getProcessNoiseQ(self, cls_label: int) -> np.ndarray:
        """set process noise(fix)"""
        if self.matrix_q is None:
            return np.asarray(np.eye(self.SD)) * 100
        return np.asarray(self.matrix_q[cls_label])

    def getMeaNoiseR(self, cls_label: int) -> np.ndarray:
        """set measure noise(fix)"""
        if self.matrix_r is None:
            return np.asarray(np.eye(self.MD)) * 0.001
        return np.asarray(self.matrix_r[cls_label])

    def getTransitionF(self, state) -> np.ndarray:
        """obtain matrix in the motion_module/script/CV_kinect_jacobian.ipynb"""
        dt = self.dt
        F = np.asarray(
            [
                [1, 0, 0, 0, 0, 0, dt, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, dt, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        return F

    def getMeaStateH(self, state=None) -> np.ndarray:
        """obtain matrix in the motion_module/script/CV_kinect_jacobian.ipynb"""
        if self.has_velo:
            H = np.asarray(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )
        else:
            H = np.asarray(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )
        return H

    def getOutputInfo(self, state: np.ndarray) -> np.array:
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        """
        rotation = Quaternion(axis=(0, 0, 1), radians=state[-1, 0]).q
        list_state = state.T.tolist()[0][:8] + rotation.tolist()
        return np.array(list_state)

    @staticmethod
    def warpResYawToPi(res: np.ndarray) -> np.ndarray:
        """warp res yaw to [-pi, pi) in place

        Args:
            res (np.ndarray): [measure dim, 1]
            res infos -> [x, y, z, w, l, h, (vx, vy, optional), ry]

        Returns:
            np.ndarray: [measure dim, 1], residual warped to [-pi, pi)
        """
        res[-1, 0] = warp_to_pi(res[-1, 0])
        return res

    @staticmethod
    def warpStateYawToPi(state: np.ndarray) -> np.ndarray:
        """warp state yaw to [-pi, pi) in place

        Args:
            state (np.ndarray): [state dim, 1]
            State vector: [x, y, z, w, l, h, vx, vy, vz, ry]

        Returns:
            np.ndarray: [state dim, 1], state after warping
        """
        state[-1, 0] = warp_to_pi(state[-1, 0])
        return state


class CTRA(ABC_MODEL):
    """Constant Acceleration and Turn Rate Motion Model
    Basic info:
        State vector: [x, y, z, w, l, h, v, a, ry, ry_rate]
        Measure vector: [x, y, z, w, l, h, (vx, vy, optional), ry]
    """

    def __init__(self, has_velo: bool, dt: float, matrix_q=None, matrix_r=None) -> None:
        super().__init__()
        self.has_velo, self.dt, self.SD = has_velo, dt, 10
        self.MD = 9 if self.has_velo else 7

    def getInitState(self, det_infos: dict) -> np.ndarray:
        """from detection init tracklet
        Acceleration and yaw(turn) rate are both set to 0. when velociy
        on X/Y-Axis are available, the combined velocity is also set to 0
        """
        init_state = np.zeros(shape=self.SD)
        det, det_box = det_infos["np_array"], det_infos["nusc_box"]

        # set x, y, z, w, l, h, (v, if velo is valid)
        init_state[:6] = det[:6]
        if self.has_velo:
            init_state[6] = np.hypot(det[6], det[7])

        # set yaw
        init_state[-2] = det_box.yaw

        # only for debug
        q = Quaternion(det[8:12].tolist())
        q = -q if q.axis[-1] < 0 else q
        assert q.radians == det_box.yaw

        return init_state[:, None]

    def getInitCovP(self, cls_label: int) -> np.ndarray:
        """init errorcov. In general, when the speed is observable,
        the CTRA model can converge quickly, but when the speed is not measurable,
        we need to carefully set the initial covariance to help the model converge
        """
        if not self.has_velo:
            cls_name = CLASS_STR_TO_SEG_CLASS[cls_label]
            vector_p = (
                CTRA_INIT_EFKP[cls_name]
                if cls_name in CTRA_INIT_EFKP
                else CTRA_INIT_EFKP["car"]
            )
        else:
            vector_p = CTRA_INIT_EFKP["car"]

        return np.asarray(np.diag(vector_p))

    def getProcessNoiseQ(self, cls_label: int) -> np.ndarray:
        """set process noise(fix)"""
        return np.asarray(np.eye(self.SD)) * 1

    def getMeaNoiseR(self, cls_label: int) -> np.ndarray:
        """set measure noise(fix)"""
        return np.asarray(np.eye(self.MD)) * 1

    def stateTransition(self, state: np.ndarray) -> np.ndarray:
        """state transition,
        obtain analytical solutions in the motion_module/script/CTRA_kinect_jacobian.ipynb
        Args:
            state (np.ndarray): [state dim, 1] the estimated state of the previous frame

        Returns:
            np.ndarray: [state dim, 1] the predict state of the current frame
        """
        assert state.shape == (10, 1), "state vector number in CTRA must equal to 10"

        dt = self.dt
        x, y, z, w, l, h, v, a, theta, omega = state.T.tolist()[0]
        yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)
        next_v, next_ry = v + a * dt, theta + omega * dt

        # corner case(tiny yaw rate), prevent divide-by-zero overflow
        if abs(omega) < 0.001:
            displacement = v * dt + a * dt**2 / 2
            predict_state = [
                x + displacement * yaw_cos,
                y + displacement * yaw_sin,
                z,
                w,
                l,
                h,
                next_v,
                a,
                next_ry,
                omega,
            ]
        else:
            ry_rate_inv_square = 1.0 / (omega * omega)
            next_yaw_sin, next_yaw_cos = np.sin(next_ry), np.cos(next_ry)
            predict_state = [
                x
                + ry_rate_inv_square
                * (
                    next_v * omega * next_yaw_sin
                    + a * next_yaw_cos
                    - v * omega * yaw_sin
                    - a * yaw_cos
                ),
                y
                + ry_rate_inv_square
                * (
                    -next_v * omega * next_yaw_cos
                    + a * next_yaw_sin
                    + v * omega * yaw_cos
                    - a * yaw_sin
                ),
                z,
                w,
                l,
                h,
                next_v,
                a,
                next_ry,
                omega,
            ]

        return np.asarray(predict_state)[:, None]

    def StateToMeasure(self, state: np.ndarray) -> np.ndarray:
        """get state vector in the measure space
        state vector -> [x, y, z, w, l, h, v, a, ry, ry_rate]
        measure space -> [x, y, z, w, l, h, (vx, vy, optional), ry]

        Args:
            state (np.ndarray): [state dim, 1] the predict state of the current frame

        Returns:
            np.ndarray: [measure dim, 1] state vector projected in the measure space
        """
        assert state.shape == (10, 1), "state vector number in CTRA must equal to 10"

        x, y, z, w, l, h, v, _, theta, _ = state.T.tolist()[0]
        if self.has_velo:
            state_info = [x, y, z, w, l, h, v * np.cos(theta), v * np.sin(theta), theta]
        else:
            state_info = [x, y, z, w, l, h, theta]

        return np.asarray(state_info)[:, None]

    def getTransitionF(self, state: np.ndarray) -> np.ndarray:
        """obtain matrix in the motion_module/script/CTRA_kinect_jacobian.ipynb
        d(stateTransition) / d(state) at previous_state
        """
        dt = self.dt
        _, _, _, _, _, _, v, a, theta, omega = state.T.tolist()[0]
        yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)

        # corner case, tiny turn rate
        if abs(omega) < 0.001:
            displacement = v * dt + a * dt**2 / 2
            F = np.asarray(
                [
                    [
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        dt * yaw_cos,
                        dt**2 * yaw_cos / 2,
                        -displacement * yaw_sin,
                        0,
                    ],
                    [
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        dt * yaw_sin,
                        dt**2 * yaw_sin / 2,
                        displacement * yaw_cos,
                        0,
                    ],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, dt, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, dt],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )
        else:
            ry_rate_inv, ry_rate_inv_square, ry_rate_inv_cube = (
                1 / omega,
                1 / (omega * omega),
                1 / (omega * omega * omega),
            )
            next_v, next_ry = v + a * dt, theta + omega * dt
            next_yaw_sin, next_yaw_cos = np.sin(next_ry), np.cos(next_ry)
            F = np.asarray(
                [
                    [
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -ry_rate_inv * (yaw_sin - next_yaw_sin),
                        -ry_rate_inv_square * (yaw_cos - next_yaw_cos)
                        + ry_rate_inv * dt * next_yaw_sin,
                        ry_rate_inv_square * a * (yaw_sin - next_yaw_sin)
                        + ry_rate_inv * (next_v * next_yaw_cos - v * yaw_cos),
                        ry_rate_inv_cube * 2 * a * (yaw_cos - next_yaw_cos)
                        + ry_rate_inv_square
                        * (v * yaw_sin - v * next_yaw_sin - 2 * a * dt * next_yaw_sin)
                        + ry_rate_inv * dt * next_v * next_yaw_cos,
                    ],
                    [
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        ry_rate_inv * (yaw_cos - next_yaw_cos),
                        -ry_rate_inv_square * (yaw_sin - next_yaw_sin)
                        - ry_rate_inv * dt * next_yaw_cos,
                        ry_rate_inv_square * a * (-yaw_cos + next_yaw_cos)
                        + ry_rate_inv * (next_v * next_yaw_sin - v * yaw_sin),
                        ry_rate_inv_cube * 2 * a * (yaw_sin - next_yaw_sin)
                        + ry_rate_inv_square
                        * (v * next_yaw_cos - v * yaw_cos + 2 * a * dt * next_yaw_cos)
                        + ry_rate_inv * dt * next_v * next_yaw_sin,
                    ],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, dt, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, dt],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )

        return F

    def getMeaStateH(self, state: np.ndarray) -> np.ndarray:
        """obtain matrix in the motion_module/script/CTRA_kinect_jacobian.ipynb
        d(StateToMeasure) / d(state) at predict_state
        """

        if self.has_velo:
            _, _, _, _, _, _, v, _, theta, _ = state.T.tolist()[0]
            yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)
            H = np.asarray(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, yaw_cos, 0, -v * yaw_sin, 0],
                    [0, 0, 0, 0, 0, 0, yaw_sin, 0, v * yaw_cos, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                ]
            )
        else:
            H = np.asarray(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                ]
            )
        return H

    def getOutputInfo(self, state: np.ndarray) -> np.array:
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        """
        rotation = Quaternion(axis=(0, 0, 1), radians=state[-2, 0]).q
        list_state = state.T.tolist()[0][:8] + rotation.tolist()
        return np.array(list_state)

    @staticmethod
    def warpResYawToPi(res: np.ndarray) -> np.ndarray:
        """warp res yaw to [-pi, pi) in place

        Args:
            res (np.ndarray): [measure dim, 1]
            res infos -> [x, y, z, w, l, h, (vx, vy, optional), ry]

        Returns:
            np.ndarray: [measure dim, 1], residual warped to [-pi, pi)
        """
        res[-1, 0] = warp_to_pi(res[-1, 0])
        return res

    @staticmethod
    def warpStateYawToPi(state: np.ndarray) -> np.ndarray:
        """warp state yaw to [-pi, pi) in place

        Args:
            state (np.ndarray): [state dim, 1]
            State vector: [x, y, z, w, l, h, v, a, ry, ry_rate]

        Returns:
            np.ndarray: [state dim, 1], state after warping
        """
        state[-2, 0] = warp_to_pi(state[-2, 0])
        return state


def get_model_class_by_name(model_name: str):
    """get model class by model name

    Args:
        model_name (str): one of ``CV``, ``CA``, or ``CTRA``

    Returns:
        class: model class
    """
    model_dict = {
        "CTRA": CTRA,
        "CV": CV,
        "CA": CA,
    }
    assert (
        model_name in model_dict
    ), f"model name {model_name} not supported, please check!"
    return model_dict[model_name]
