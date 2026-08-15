// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// cpu_engine: portable CPU executor for the same plugin-agnostic IR consumed by
// the Vulkan runtime (vk_program/vk_network). It runs the graph entirely on the
// host (f32 arithmetic) and dequantizes quantized weight constants natively
// with the exact formulas used by matmul_q_f32.comp -- no GPU required. This is
// the cross-platform fallback / reference backend of the standalone core.

#pragma once

#include "vk_ir.hpp"

#include <map>
#include <string>
#include <vector>

namespace ov::core {
namespace vulkan {
namespace cross_platform {

// Runs |g| (topologically sorted) with the given model inputs and returns a map
// of output buffer id -> f32 payload for every id in g.outputs.
std::map<std::string, std::vector<float>> cpu_execute(
    const ir_graph& g, const std::map<std::string, std::vector<float>>& inputs);

// Dequantizes a raw quantized [rows, cols] weight tensor (row-major, blocks
// along cols) into f32. Uses the same block layouts as matmul_q_f32.comp.
std::vector<float> cpu_dequant(const ir_quant_const& qc, size_t rows, size_t cols);

}  // namespace cross_platform
}  // namespace vulkan
}  // namespace ov::core
