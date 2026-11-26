// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "config.h"
#include "executor_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/op/gated_residual.hpp"

namespace ov::intel_cpu {

struct GatedResidualAttrs {
    size_t hc_count = 0;
    size_t hidden_size = 0;
    size_t lowrank = 0;
    float epsilon = 1e-6F;
    bool produce_write_gate = false;
    // Input form of the node; see GatedResidual::ReadLayout.
    ov::op::internal::GatedResidual::ReadLayout read_layout{};
    // FullyConnected settings of the projections, taken from the plugin config as the FullyConnected node does.
    uint64_t dynamic_quantization_group_size = 0;
    ov::intel_cpu::Config::ModelType model_type = ov::intel_cpu::Config::ModelType::Unknown;
    // Contexts of the down, up and inject projections, each with its own private weight cache (keyed by layout only).
    std::array<ExecutorContext::CPtr, 3> projection_contexts{};
};

using GatedResidualConfig = executor::Config<GatedResidualAttrs>;

// Children of the GatedResidual executor's scratchpad. Rbar uses the pad's own block. The projections run one after
// another, so they share one child.
enum GatedResidualPad : uint8_t { kPadLogit, kPadT, kPadWriteGate, kPadRbarGemm, kPadProjections };

// Node port i maps to ARG_SRC_0 + i (inputs) and ARG_DST_0 + i (outputs).
constexpr int gatedResidualSrcArg(size_t port) {
    return ARG_SRC_0 + static_cast<int>(port);
}

constexpr int gatedResidualDstArg(size_t port) {
    return ARG_DST_0 + static_cast<int>(port);
}

}  // namespace ov::intel_cpu
