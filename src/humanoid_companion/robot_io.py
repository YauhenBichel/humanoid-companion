"""How the control loop talks to a robot: read sensors, write joint targets, make safe.

MujocoRobotIO is plain CPU MuJoCo with the model the policy was trained on (Playground's
env.mj_model, which already carries the training PD gains and timestep). DynamixelRobotIO is
the real OP3-class robot, still to be written (it needs the hardware and a system identification).
"""

from typing import Protocol

import mujoco
import numpy as np

from humanoid_companion.observation import Sensors


class RobotIO(Protocol):
    def read(self) -> Sensors: ...
    def write(self, joint_targets: np.ndarray) -> None: ...
    def stop(self) -> None: ...


class MujocoRobotIO:
    """Simulated robot. `write` applies the targets and advances one control period."""

    def __init__(self, model: mujoco.MjModel, ctrl_dt: float, keyframe: str = "stand_bent_knees",
                 gyro_sensor: str = "gyro", up_sensor: str = "upvector", root_body: str = "body_link"):
        self.model = model
        self.data = mujoco.MjData(model)
        self.n_substeps = int(round(ctrl_dt / model.opt.timestep))
        self._gyro = self._sensor_slice(gyro_sensor)
        self._up = self._sensor_slice(up_sensor)
        self._root = model.body(root_body).id
        self.reset(keyframe)

    def _sensor_slice(self, name: str) -> slice:
        s = self.model.sensor(name)
        return slice(int(s.adr[0]), int(s.adr[0] + s.dim[0]))

    def reset(self, keyframe: str = "stand_bent_knees") -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.keyframe(keyframe).id)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.data.qpos[7:]
        mujoco.mj_forward(self.model, self.data)

    def read(self) -> Sensors:
        d = self.data
        return Sensors(gyro=d.sensordata[self._gyro].copy(), upvector=d.sensordata[self._up].copy(),
                       joint_pos=d.qpos[7:].copy(), time=float(d.time))

    def write(self, joint_targets: np.ndarray) -> None:
        self.data.ctrl[:] = joint_targets
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

    def stop(self) -> None:
        """Nothing to power down in simulation; the loop stops stepping."""

    @property
    def base_position(self) -> np.ndarray:
        return self.data.xpos[self._root].copy()


class DynamixelRobotIO:
    """The real robot (Dynamixel servos + IMU): not written yet."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DynamixelRobotIO is not written yet: it needs the hardware and a system identification")
