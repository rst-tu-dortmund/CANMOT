"""
kalman filter for trajectory state(motion state) estimation
Two implemented KF version (LKF, EKF)
Three core functions for each model: state init, state predict and state update
Linear Kalman Filter for CA and CV; extended Kalman filter for CTRA
Ref: https://en.wikipedia.org/wiki/Kalman_filter
"""

import pdb
import logging
import numpy as np
from .nusc_object import FrameObject
from . import motion_model
from canmot.pre_processing import arraydet2box, concat_box_attr
from pyquaternion import Quaternion

logger = logging.getLogger(__name__)

# Per-class dedup guard: only log Q/R/P details for the first track of each (filter_class, class_label)
_logged_classes = set()


class KalmanFilter:
    """kalman filter interface"""

    def __init__(
        self, timestamp: int, config: dict, track_id: int, det_infos: dict
    ) -> None:
        # init basic infos, no control input
        self.seq_id = det_infos["seq_id"]
        self.initstamp = self.timestamp = timestamp
        self.tracking_id, self.class_label = track_id, int(det_infos["np_array"][-1])
        model_name = config["motion_model"]["model"][self.class_label]
        self.model_class = motion_model.get_model_class_by_name(model_name)
        self.model = None
        self.dt, self.has_velo = (
            config["basic"]["LiDAR_interval"],
            config["basic"]["has_velo"],
        )
        # init FrameObject for each frame
        (
            self.state,
            self.innovation_information,
            self.state_meas_space,
            self.frame_objects,
        ) = (None, None, None, {})

    def initialize(self, det: dict) -> None:
        """initialize the filter parameters
        Args:
            det (dict): detection infos under different data format.
            {
                'nusc_box': NuscBox,
                'np_array': np.array,
                'has_velo': bool, whether the detetor has velocity info
            }
        """
        pass

    def predict(self, timestamp: int) -> None:
        """predict tracklet at each frame
        Args:
            timestamp (int): current frame id
        """
        pass

    def update(self, timestamp: int, det: dict = None) -> None:
        """update tracklet motion and geometric state
        Args:
            timestamp (int): current frame id
            det (dict, optional): same as self.init. Defaults to None.
        """
        pass

    def getMeasureInfo(self, det: dict = None) -> np.array:
        """convert det box to the measurement info for updating filter
        [x, y, z, w, h, l, (vx, vy, optional), ry]
        Args:
            det (dict, optional): same as self.init. Defaults to None.

        Returns:
            np.array: measurement for updating filter
        """
        if det is None:
            raise "detection cannot be None"

        mea_attr = (
            ("center", "wlh", "velocity", "yaw")
            if self.has_velo
            else ("center", "wlh", "yaw")
        )
        list_det = concat_box_attr(det["nusc_box"], *mea_attr)
        if self.has_velo:
            list_det.pop(8)

        # Ensure the measurement yaw uses the expected quaternion axis.
        assert (
            list_det[-1] == det["nusc_box"].orientation.radians
            and det["nusc_box"].orientation.axis[-1] >= 0
        )
        assert len(list_det) == 9 if self.has_velo else 7

        return np.asarray(list_det)[:, None]

    def addFrameObject(self, timestamp: int, tra_info: dict, mode: str = None) -> None:
        """add predict/update tracklet state to the frameobjects, data
        format is also implemented in this function.
        frame_objects: {
            frame_id: FrameObject
        }
        Args:
            timestamp (int): current frame id
            tra_info (dict): Trajectory state estimated by Kalman filter,
            {
                'exter_state': np.array, for output file.
                               [x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), tra_score, class_label]
                'inner_state': np.array, for state estimation.
                               varies by motion model
            }
            mode (str, optional): stage of adding objects, 'update', 'predict'. Defaults to None.
        """
        # corner case, no tra info
        if mode is None:
            return

        # data format conversion
        inner_info, exter_info = tra_info["inner_state"], tra_info["exter_state"]
        extra_info = np.asarray([self.tracking_id, self.seq_id, timestamp])
        box_info, bm_info = arraydet2box(exter_info, np.asarray([self.tracking_id]))
        if len(box_info) > 0 and hasattr(self, "P") and self.P is not None:
            box_info[0].covariance = np.asarray(self.get_covariance_meas(), dtype=float).tolist()

        # update each frame infos
        if mode == "update":
            frame_object = self.frame_objects[timestamp]
            frame_object.update_bms, frame_object.update_box = bm_info[0], box_info[0]
            frame_object.update_state, frame_object.update_infos = (
                inner_info,
                np.append(exter_info, extra_info),
            )
        elif mode == "predict":
            frame_object = FrameObject()
            frame_object.predict_bms, frame_object.predict_box = bm_info[0], box_info[0]
            frame_object.predict_state, frame_object.predict_infos = (
                inner_info,
                np.append(exter_info, extra_info),
            )
            self.frame_objects[timestamp] = frame_object
        else:
            raise Exception("mode must be update or predict")

    def get_covariance_meas(self) -> np.array:
        raise NotImplementedError("Subclasses must implement measurement covariance export")

    def get_measurement_noise_covariance(self) -> np.array:
        raise NotImplementedError("Subclasses must implement measurement noise covariance export")

    def getOutputInfo(self, state: np.array) -> np.array:
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        Args:
            state (np.mat): [state dim, 1], predict or update state estimated by the filter

        Returns:
            np.array: [14(fix), 1], predict or update state under output file format
            output format: [x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), tra_score, class_label]
        """

        # return state vector except tra score and tra class
        inner_state = self.model.getOutputInfo(state)
        return np.append(inner_state, np.asarray([-1, self.class_label]))

    def __getitem__(self, item) -> FrameObject:
        return self.frame_objects[item]

    def __len__(self) -> int:
        return len(self.frame_objects)


class LinearKalmanFilter(KalmanFilter):
    """Linear Kalman Filter for linear motion model, such as CV and CA"""

    def __init__(
        self, timestamp: int, config: dict, track_id: int, det_infos: dict
    ) -> None:
        # init basic infos
        super(LinearKalmanFilter, self).__init__(timestamp, config, track_id, det_infos)
        # set motion model, default Constant Acceleration(CA) for LKF
        self.model = self.model_class(
            self.has_velo,
            self.dt,
            config["model_cfg"]["matrix_q"],
            config["model_cfg"]["matrix_r"],
        )
        # Transition and Observation Matrices are fixed in the LKF
        self.initialize(det_infos)

    def initialize(self, det_infos: dict) -> None:
        # state transition
        self.F = self.model.getTransitionF(
            None
        )  # for linear model, F is fixed and not related to state
        self.Q = self.model.getProcessNoiseQ(self.class_label)
        self.SD = self.model.getStateDim()
        self.P = self.model.getInitCovP(self.class_label)

        # state to measurement transition
        self.R = self.model.getMeaNoiseR(self.class_label)
        self.H = self.model.getMeaStateH()

        # Diagnostic logging (once per class)
        _key = ("LinearKalmanFilter", int(self.class_label))
        if _key not in _logged_classes:
            _logged_classes.add(_key)
            logger.info("[LinearKalmanFilter] Class %d: Q shape=%s, Q diag=%s",
                        self.class_label, self.Q.shape, np.diag(self.Q))
            logger.info("[LinearKalmanFilter] Class %d: R shape=%s, R diag=%s",
                        self.class_label, self.R.shape, np.diag(self.R))
            logger.info("[LinearKalmanFilter] Class %d: P_init diag=%s",
                        self.class_label, np.diag(self.P))
            logger.info("[LinearKalmanFilter] Class %d: Q is symmetric=%s, R is symmetric=%s",
                        self.class_label,
                        np.allclose(self.Q, self.Q.T),
                        np.allclose(self.R, self.R.T))
            logger.info("[LinearKalmanFilter] Class %d: NOTE — Q and R are used directly (global frame, no rotation)",
                        self.class_label)

        # get different data format tracklet's state
        self.state = self.model.getInitState(det_infos)
        tra_infos = {"inner_state": self.state, "exter_state": det_infos["np_array"]}
        self.addFrameObject(self.timestamp, tra_infos, "predict")
        self.addFrameObject(self.timestamp, tra_infos, "update")

    def predict(self, timestamp: int) -> None:
        # predict state and errorcov
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.innovation_information = np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.state_meas_space = self.H @ self.state

        # convert the state in filter to the output format
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {"inner_state": self.state, "exter_state": output_info}
        self.addFrameObject(timestamp, tra_infos, "predict")

    def update(self, timestamp: int, det: dict = None) -> None:
        # corner case, no det for updating
        if det is None:
            return

        # only for debug
        q = Quaternion(det["np_array"][8:12].tolist())
        q = -q if q.axis[-1] < 0 else q
        assert q.radians == det["nusc_box"].yaw

        # update state and errorcov
        meas_info = self.getMeasureInfo(det)
        _res = meas_info - self.state_meas_space
        self.model.warpResYawToPi(_res)
        # _S = self.H @ self.P @ self.H.T + self.R
        _KF_GAIN = self.P @ self.H.T @ self.innovation_information

        self.state += _KF_GAIN @ _res
        self.P = (np.asarray(np.eye(self.SD)) - _KF_GAIN @ self.H) @ self.P

        # output updated state to the result file
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {"inner_state": self.state, "exter_state": output_info}
        self.addFrameObject(timestamp, tra_infos, "update")

    def get_covariance_meas(self) -> np.array:
        """get the covariance matrix in the measurement space, for outputting to file
        Returns:
            np.array: covariance matrix of the current state in measurement space, with shape [state_dim, state_dim]
        """
        return self.H @ self.P @ self.H.T

    def get_measurement_noise_covariance(self) -> np.array:
        """measurement noise covariance in measurement space"""
        return self.R


class ExtendKalmanFilter(KalmanFilter):
    def __init__(
        self, timestamp: int, config: dict, track_id: int, det_infos: dict
    ) -> None:
        super().__init__(timestamp, config, track_id, det_infos)
        # set motion model, default Constant Acceleration and Turn Rate(CTRA) for EKF
        self.model = self.model_class(
            self.has_velo,
            self.dt,
            config["model_cfg"]["matrix_q"],
            config["model_cfg"]["matrix_r"],
        )
        # Transition and Observation Matrices are changing in the EKF
        self.initialize(det_infos)

    def initialize(self, det_infos: dict) -> None:
        # init errorcov categoty-specific
        self.SD = self.model.getStateDim()
        self.P = self.model.getInitCovP(self.class_label)

        # set noise matrix(fixed)
        self.Q = self.model.getProcessNoiseQ(self.class_label)
        self.R = self.model.getMeaNoiseR(self.class_label)

        # get different data format tracklet's state
        self.state = self.model.getInitState(det_infos)
        tra_infos = {"inner_state": self.state, "exter_state": det_infos["np_array"]}
        self.addFrameObject(self.timestamp, tra_infos, "predict")
        self.addFrameObject(self.timestamp, tra_infos, "update")

    def predict(self, timestamp: int) -> None:
        # get jacobian matrix F using the final estimated state of the previous frame
        self.F = self.model.getTransitionF(self.state)

        # state and errorcov transition
        self.state = self.model.stateTransition(self.state)
        self.P = self.F @ self.P @ self.F.T + self.Q

        # precompute information covariance matrix and state in measurement space
        self.H = self.model.getMeaStateH(self.state)
        self.innovation_information = np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.state_meas_space = self.model.StateToMeasure(self.state)

        # convert the state in filter to the output format
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {"inner_state": self.state, "exter_state": output_info}
        self.addFrameObject(timestamp, tra_infos, "predict")

    def update(self, timestamp: int, det: dict = None) -> None:
        # corner case, no det for updating
        if det is None:
            return

        # only for debug
        q = Quaternion(det["np_array"][8:12].tolist())
        q = -q if q.axis[-1] < 0 else q
        assert q.radians == det["nusc_box"].yaw

        # get measure infos for updating, and project state into meausre space
        meas_info = self.getMeasureInfo(det)
        state_info = self.state_meas_space

        # get state residual, and warp angle diff inplace
        _res = meas_info - state_info
        self.model.warpResYawToPi(_res)

        # get jacobian matrix H using the predict state
        # self.H = self.model.getMeaStateH(self.state) # already calculated in prediction

        # obtain KF gain and update state and errorcov
        # _S = self.H @ self.P @ self.H.T + self.R
        _KF_GAIN = self.P @ self.H.T @ self.innovation_information
        _I_KH = np.asarray(np.eye(self.SD)) - _KF_GAIN @ self.H
        resid = _KF_GAIN @ _res
        self.state += resid
        self.P = _I_KH @ self.P @ _I_KH.T + _KF_GAIN @ self.R @ _KF_GAIN.T

        # output updated state to the result file
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {"inner_state": self.state, "exter_state": output_info}
        self.addFrameObject(timestamp, tra_infos, "update")

    def get_covariance_meas(self) -> np.array:
        """get the covariance matrix in the measurement space, for outputting to file
        Returns:
            np.array: covariance matrix of the current state in measurement space, with shape [state_dim, state_dim]
        """
        H = self.model.getMeaStateH(self.state)  # get current H based on updated state
        return H @ self.P @ H.T

    def get_measurement_noise_covariance(self) -> np.array:
        """measurement noise covariance in measurement space"""
        return self.R


class ExtendedKalmanFilterRotate(KalmanFilter):
    def __init__(
        self, timestamp: int, config: dict, track_id: int, det_infos: dict
    ) -> None:
        super().__init__(timestamp, config, track_id, det_infos)
        # all models (even linear ones) are allowed, as filter will be linear if linear model is used
        self.model = self.model_class(
            self.has_velo,
            self.dt,
            config["model_cfg"]["matrix_q"],
            config["model_cfg"]["matrix_r"],
        )
        # Transition and Observation Matrices are changing in the EKF
        self._first_predict = True  # flag for one-time predict logging
        self.initialize(det_infos)

    def initialize(self, det_infos: dict) -> None:
        # init errorcov category-specific
        self.SD = self.model.getStateDim()
        # self.P = self.model.getInitCovP(self.class_label)

        # set noise matrix(fixed) — these are in LOCAL (object) frame
        self.Q = self.model.getProcessNoiseQ(self.class_label)
        self.R = self.model.getMeaNoiseR(self.class_label)

        # get different data format tracklet's state
        self.state = self.model.getInitState(det_infos)
        init_rotmat = self.model.get_rotation_matrix_for_measurement(self.state)
        H = self.model.getMeaStateH(self.state)
        self.P = H.T @ init_rotmat @ self.R @ init_rotmat.T @ H  # initialize P by rotating R according to initial state
        self.P[8, 8] = 10.0  # vz is unobserved, set a high initial variance so it can be updated
        tra_infos = {"inner_state": self.state, "exter_state": det_infos["np_array"]}
        self.addFrameObject(self.timestamp, tra_infos, "predict")
        self.addFrameObject(self.timestamp, tra_infos, "update")

        # Diagnostic logging (once per class)
        _key = ("ExtendedKalmanFilterRotate", int(self.class_label))
        if _key not in _logged_classes:
            _logged_classes.add(_key)
            logger.info("[EKFRotate] Class %d: Q shape=%s, Q diag (LOCAL frame)=%s",
                        self.class_label, self.Q.shape, np.diag(self.Q))
            logger.info("[EKFRotate] Class %d: R shape=%s, R diag (LOCAL frame)=%s",
                        self.class_label, self.R.shape, np.diag(self.R))
            logger.info("[EKFRotate] Class %d: P_init diag=%s",
                        self.class_label, np.diag(self.P))
            logger.info("[EKFRotate] Class %d: Q is symmetric=%s, R is symmetric=%s",
                        self.class_label,
                        np.allclose(self.Q, self.Q.T),
                        np.allclose(self.R, self.R.T))
            logger.info("[EKFRotate] Class %d: NOTE — Q and R will be ROTATED by yaw at each predict/update",
                        self.class_label)

    def predict(self, timestamp: int) -> None:
        # get jacobian matrix F using the final estimated state of the previous frame
        self.F = self.model.getTransitionF(self.state)

        # state and errorcov transition
        rotmat_prev = self.model.get_rotation_matrix_for_state(self.state)
        self.state = self.model.stateTransition(self.state)
        rotmat_post_meas = self.model.get_rotation_matrix_for_measurement(self.state)
        Q = rotmat_prev @ self.Q @ rotmat_prev.T  # rotate process noise covariance
        self.P = self.F @ self.P @ self.F.T + Q

        # precompute measurement space
        self.H = self.model.getMeaStateH(self.state)
        R_rotated = rotmat_post_meas @ self.R @ rotmat_post_meas.T
        self.innovation_information = np.linalg.inv(
            self.H @ self.P @ self.H.T + R_rotated
        )
        self.state_meas_space = self.model.StateToMeasure(self.state)

        # One-time diagnostic logging for the first predict call of this track
        if self._first_predict:
            self._first_predict = False
            yaw = self.state[-1, 0]
            logger.debug("[EKFRotate] First predict — class %d, track %d, yaw=%.4f rad (%.1f deg)",
                         self.class_label, self.tracking_id, yaw, np.degrees(yaw))
            logger.debug("[EKFRotate]   Q_local diag: %s", np.diag(self.Q))
            logger.debug("[EKFRotate]   Q_rotated diag: %s", np.diag(Q))
            logger.debug("[EKFRotate]   R_local diag: %s", np.diag(self.R))
            logger.debug("[EKFRotate]   R_rotated diag: %s", np.diag(R_rotated))
            logger.debug("[EKFRotate]   Q_rotated symmetric=%s, PSD=%s",
                         np.allclose(Q, Q.T),
                         np.all(np.linalg.eigvalsh(Q) >= -1e-10))
            logger.debug("[EKFRotate]   R_rotated symmetric=%s, PSD=%s",
                         np.allclose(R_rotated, R_rotated.T),
                         np.all(np.linalg.eigvalsh(R_rotated) >= -1e-10))

        # convert the state in filter to the output format
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {"inner_state": self.state, "exter_state": output_info}
        self.addFrameObject(timestamp, tra_infos, "predict")

    def update(self, timestamp: int, det: dict = None) -> None:
        # corner case, no det for updating
        if det is None:
            return

        # only for debug
        q = Quaternion(det["np_array"][8:12].tolist())
        q = -q if q.axis[-1] < 0 else q
        assert q.radians == det["nusc_box"].yaw

        # get measure infos for updating, and project state into meausre space
        meas_info = self.getMeasureInfo(det)
        state_info = self.state_meas_space

        # get state residual, and warp angle diff inplace
        _res = meas_info - state_info
        self.model.warpResYawToPi(_res)

        # get jacobian matrix H using the predict state
        # self.H = self.model.getMeaStateH(self.state) # already calculated in prediction

        # obtain KF gain and update state and errorcov
        rotmat = self.model.get_rotation_matrix_for_measurement(self.state)
        R = rotmat @ self.R @ rotmat.T  # rotate measurement noise covariance
        # _S = self.H @ self.P @ self.H.T + R
        _KF_GAIN = self.P @ self.H.T @ self.innovation_information
        _I_KH = np.asarray(np.eye(self.SD)) - _KF_GAIN @ self.H
        resid = _KF_GAIN @ _res
        self.state += resid
        self.P = _I_KH @ self.P @ _I_KH.T + _KF_GAIN @ R @ _KF_GAIN.T

        # output updated state to the result file
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {"inner_state": self.state, "exter_state": output_info}
        self.addFrameObject(timestamp, tra_infos, "update")

    def get_covariance_meas(self) -> np.array:
        """get the covariance matrix in the measurement space, for outputting to file
        Returns:
            np.array: covariance matrix of the current state in measurement space, with shape [state_dim, state_dim]
        """
        # no rotation required - P is already in global frame
        H = self.model.getMeaStateH(self.state)  # get current H based on updated state
        return H @ self.P @ H.T

    def get_measurement_noise_covariance(self) -> np.array:
        """measurement noise covariance in measurement space after yaw rotation"""
        rotmat = self.model.get_rotation_matrix_for_measurement(self.state)
        return rotmat @ self.R @ rotmat.T
