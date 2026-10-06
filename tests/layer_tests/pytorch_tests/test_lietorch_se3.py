# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.autograd.function import Function

from pytorch_layer_test_class import PytorchLayerTest


def _cross_last(a, b):
    # cross product along the last axis
    ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3]
    bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3]
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return torch.cat([cx, cy, cz], dim=-1)


def _se3_exp(tangent):
    # tangent (..., 6) -> SE3 data (..., 7) = [tx, ty, tz, qx, qy, qz, qw]
    tau = tangent[..., 0:3]
    phi = tangent[..., 3:6]
    theta2 = (phi * phi).sum(-1, keepdim=True) + 1e-12
    theta = torch.sqrt(theta2)
    half_theta = theta * 0.5
    qw = torch.cos(half_theta)
    imag = torch.sin(half_theta) / theta
    q = torch.cat([imag * phi, qw], dim=-1)
    c1 = (1.0 - torch.cos(theta)) / theta2
    c2 = (theta - torch.sin(theta)) / (theta2 * theta)
    phi_x_tau = _cross_last(phi, tau)
    phi_x_phi_x_tau = _cross_last(phi, phi_x_tau)
    t = tau + c1 * phi_x_tau + c2 * phi_x_phi_x_tau
    return torch.cat([t, q], dim=-1)


def _se3_act(data, points):
    # data (..., 7), points (..., 3) -> (..., 3)
    t = data[..., 0:3]
    qv = data[..., 3:6]
    qw = data[..., 6:7]
    u = 2.0 * _cross_last(qv, points)
    p_rot = (points + qw * u) + _cross_last(qv, u)
    return p_rot + t


class Exp(Function):
    @staticmethod
    def forward(ctx, tangent):
        ctx.save_for_backward(tangent)
        return _se3_exp(tangent)


class Act3(Function):
    @staticmethod
    def forward(ctx, data, points):
        ctx.save_for_backward(data, points)
        return _se3_act(data, points)


Exp.__module__ = "lietorch.group_ops"
Act3.__module__ = "lietorch.group_ops"


class TestLieTorchSE3Exp(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 6, dtype="float32"),)

    def create_model(self):
        class lietorch_exp(torch.nn.Module):
            def forward(self, tangent):
                # the reshape gives the op a static rank input, like lietorch does
                return Exp.apply(tangent.reshape(-1, 6))

        return lietorch_exp(), "prim::PythonOp"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_lietorch_se3_exp(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   trace_model=True)


class TestLieTorchSE3Act(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 8, 7, dtype="float32"),
                self.random.randn(2, 8, 3, dtype="float32"))

    def create_model(self):
        class lietorch_act(torch.nn.Module):
            def forward(self, data, points):
                d = data.reshape(data.shape[0], -1, 7)
                p = points.reshape(points.shape[0], -1, 3)
                return Act3.apply(d, p)

        return lietorch_act(), "prim::PythonOp"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_lietorch_se3_act(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   trace_model=True)
