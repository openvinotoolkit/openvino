// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// paddle_reader: autonomous PaddlePaddle -> ir_graph loader for the standalone
// core. Parses a serialized ProgramDesc (.pdmodel, protobuf wire format, no
// protobuf library) plus the per-parameter LoDTensor files and produces the
// same ir_graph the Vulkan runtime and the CPU engine consume -- no ov::Model,
// no openvino core. Mirrors the gguf_reader pattern.
//
// Supported op subset (NCHW, f32 weights):
//   feed / fetch (I/O), relu, elementwise_add (same shape),
//   matmul (+transpose_Y), conv2d (groups=1, NCHW), max_pool2d / avg_pool2d
//   (floor padding, avg exclude_pad=true). Anything else -> clean error.

#pragma once

#include "vk_ir.hpp"

#include <functional>
#include <string>
#include <vector>

namespace ov::core {
namespace vulkan {
namespace cross_platform {
namespace paddle_r {

// Reads an inference-model directory: <dir>/__model__ (ProgramDesc) + one
// LoDTensor file per persistable weight (<dir>/<var_name>).
ir_graph paddle_load_model(const std::string& dir);

// Testable core: parses a serialized ProgramDesc, fetching each persistable
// weight's raw bytes through |load_param| (called with the var name).
ir_graph paddle_parse_program(
    const std::vector<uint8_t>& program,
    const std::function<std::vector<uint8_t>(const std::string& var_name)>& load_param);

}  // namespace paddle_r
}  // namespace cross_platform
}  // namespace vulkan
}  // namespace ov::core
