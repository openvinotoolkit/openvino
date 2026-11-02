// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "executor_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/op/gated_residual.hpp"

namespace ov::intel_cpu {

struct GatedResidualAttrs {
    ov::op::internal::GatedResidual::Mode mode = ov::op::internal::GatedResidual::Mode::READ;
    size_t hc_count = 0;
    size_t hidden_size = 0;
    size_t lowrank = 0;
    float epsilon = 1e-6F;
    bool norm_gain_plus_one = true;
    bool produce_write_gate = false;
    // w_down/w_up/(w_inject) are u8/i8/u4/i4, with f32 scale (and optionally zero-point) ports; see
    // ov::op::internal::GatedResidual::CompressedInputIdx for the port layout.
    bool weights_compressed = false;
    bool weights_have_zero_points = false;
    size_t weight_group_size = 0;
};

using GatedResidualConfig = executor::Config<GatedResidualAttrs>;

// The op has up to 11 inputs (READ with write gate and zero-points) and 2 outputs; each node port
// maps 1:1 onto ARG_SRC_0 + port / ARG_DST_0 + port.
constexpr int gatedResidualSrcArg(size_t port) {
    return ARG_SRC_0 + static_cast<int>(port);
}

constexpr int gatedResidualDstArg(size_t port) {
    return ARG_DST_0 + static_cast<int>(port);
}

}  // namespace ov::intel_cpu
