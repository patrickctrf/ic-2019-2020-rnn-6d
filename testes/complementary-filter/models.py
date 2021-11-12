import torch
from torch import nn

from ptk.utils.torchtools import axis_angle_into_rotation_matrix, axis_angle_into_quaternion, rotation_matrix_into_axis_angle


class EachSamplePreintegrationModule(nn.Module):
    def __init__(self, device=torch.device("cpu"), dtype=torch.float32, imu_freq=200):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.delta_t = 1 / imu_freq

        self.identity_matrix = torch.eye(n=3, m=3, device=device, dtype=dtype)

    def forward(self, input_seq):
        """
    This method computes delta R, v and p (orientation, velocity and position),
    according to https://arxiv.org/abs/2101.07061 and
    https://arxiv.org/abs/1512.02363 about inertial feature preintegration.

        :param a_samples: IMU accelerometer input samples to compute over.
        :param w_samples: IMU gyroscope input samples to compute over.
        :param initial_velocity: Whenever using a statefull approach, passing
        initial_velocity will bring up better preintegration results.
        :return: R (converted from matriz into quaternion), v and p (both 3D tensors).
        """
        # orientation matrix
        delta_r = self.identity_matrix.clone()
        # velocity tensor (3 element)
        delta_v = 0.0
        # position tensor (3 element)
        delta_p = 0.0

        deltas_p = torch.zeros((input_seq.shape[0], input_seq.shape[1], 3, 1), device=self.device, dtype=self.dtype)
        deltas_v = torch.zeros((input_seq.shape[0], input_seq.shape[1], 3, 1), device=self.device, dtype=self.dtype)
        deltas_q = torch.zeros((input_seq.shape[0], input_seq.shape[1], 4), device=self.device, dtype=self.dtype)

        # avoid dividing delta_t by 2 on every loop iteration
        delta_t_divided_by_2 = self.delta_t / 2
        square_delta_t_divided_by_2 = delta_t_divided_by_2 * self.delta_t

        w = input_seq.movedim(1, 0)[:, :, :3]
        a = input_seq.movedim(1, 0)[:, :, 3:].unsqueeze(3)
        # interactive productory and summation steps
        for i, (w_k, a_k) in enumerate(list(zip(w, a))):
            delta_r = torch.matmul(delta_r, axis_angle_into_rotation_matrix(w_k, self.delta_t, device=self.device, dtype=self.dtype))
            delta_v += torch.matmul(delta_r, a_k * self.delta_t)
            # Slightly different from original paper, now including
            # initial_velocity (if available) to compute CURRENT velocity, not
            # only delta_v (variation)
            delta_p += delta_v * self.delta_t + torch.matmul(delta_r, a_k * square_delta_t_divided_by_2)

            deltas_p[:, i, :, 0:] = delta_p
            deltas_v[:, i, :, 0:] = delta_v
            deltas_q[:, i, :] = \
                axis_angle_into_quaternion(
                    *rotation_matrix_into_axis_angle(
                        delta_r, device=self.device, dtype=self.dtype
                    ), device=self.device, dtype=self.dtype
                )

        # noinspection PyTypeChecker
        return torch.cat((input_seq, deltas_p.squeeze(3), deltas_q,
                          deltas_v.squeeze(3)), dim=2)


class PreintegrationModule(nn.Module):
    def __init__(self, device=torch.device("cpu"), dtype=torch.float32, imu_freq=200):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.delta_t = 1 / imu_freq

        self.identity_matrix = torch.eye(n=3, m=3, device=device, dtype=dtype)

    def forward(self, input_seq):
        """
    This method computes delta R, v and p (orientation, velocity and position),
    according to https://arxiv.org/abs/2101.07061 and
    https://arxiv.org/abs/1512.02363 about inertial feature preintegration.

        :param a_samples: IMU accelerometer input samples to compute over.
        :param w_samples: IMU gyroscope input samples to compute over.
        :param initial_velocity: Whenever using a statefull approach, passing
        initial_velocity will bring up better preintegration results.
        :return: R (converted from matriz into quaternion), v and p (both 3D tensors).
        """
        # orientation matrix
        delta_r = self.identity_matrix.clone()
        # velocity tensor (3 element)
        delta_v = 0.0
        # position tensor (3 element)
        delta_p = 0.0

        # avoid dividing delta_t by 2 on every loop iteration
        delta_t_divided_by_2 = self.delta_t / 2
        square_delta_t_divided_by_2 = delta_t_divided_by_2 * self.delta_t

        w = input_seq.movedim(1, 0)[:, :, :3]
        a = input_seq.movedim(1, 0)[:, :, 3:].unsqueeze(3)
        # interactive productory and summation steps
        for w_k, a_k in zip(w, a):
            delta_r = torch.matmul(delta_r, axis_angle_into_rotation_matrix(w_k, self.delta_t, device=self.device, dtype=self.dtype))
            delta_v += torch.matmul(delta_r, a_k * self.delta_t)
            # Slightly different from original paper, now including
            # initial_velocity (if available) to compute CURRENT velocity, not
            # only delta_v (variation)
            delta_p += delta_v * self.delta_t + torch.matmul(delta_r, a_k * square_delta_t_divided_by_2)

        # noinspection PyTypeChecker
        return torch.cat((delta_p.squeeze(2), axis_angle_into_quaternion(*rotation_matrix_into_axis_angle(delta_r, device=self.device, dtype=self.dtype), device=self.device, dtype=self.dtype)),
                         dim=1)  # , delta_v


class SingleSamplePreintegrationModule(nn.Module):
    def __init__(self, device=torch.device("cpu"), dtype=torch.float32, imu_freq=200):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.delta_t = 1 / imu_freq

        self.identity_matrix = torch.eye(n=3, m=3, device=device, dtype=dtype)

    def forward(self, input_seq):
        """
    This method computes delta R, v and p (orientation, velocity and position),
    according to https://arxiv.org/abs/2101.07061 and
    https://arxiv.org/abs/1512.02363 about inertial feature preintegration.

        :param a_samples: IMU accelerometer input samples to compute over.
        :param w_samples: IMU gyroscope input samples to compute over.
        :param initial_velocity: Whenever using a statefull approach, passing
        initial_velocity will bring up better preintegration results.
        :return: R (converted from matriz into quaternion), v and p (both 3D tensors).
        """

        # w, a
        # input_seq[0, :, :3], input_seq[0, :, 3:]

        # single sample productory and summation steps
        delta_r = axis_angle_into_rotation_matrix(input_seq[0, :, :3], self.delta_t, device=self.device, dtype=self.dtype)
        delta_v = torch.matmul(delta_r, input_seq[0, :, 3:] * self.delta_t)
        delta_p = delta_v * self.delta_t + torch.matmul(delta_r, input_seq[0, :, 3:] * (self.delta_t ** 2) / 2)

        # noinspection PyTypeChecker
        return torch.cat((delta_p, axis_angle_into_quaternion(*rotation_matrix_into_axis_angle(delta_r, device=self.device, dtype=self.dtype), device=self.device, dtype=self.dtype), delta_v),
                         dim=-1)
