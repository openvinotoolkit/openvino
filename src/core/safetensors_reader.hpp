// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// st_r: Safetensors reader for the standalone Vulkan core.
//
// Format (https://github.com/huggingface/safetensors): the file starts with
// a little-endian u64 header size N, followed by N bytes of JSON describing
// every tensor (dtype, shape, byte offsets into the data block), followed by
// the raw data. The data block starts at 8 + N and each tensor begins at an
// 8-byte aligned offset by construction.
//
// Supported dtypes: F32, F16, BF16 (converted to f32). Anything else fails
// with a clean error — no silent reinterpretation.
//
// The JSON parser is a minimal scanner for this fixed schema (an object of
// tensor objects with dtype/shape/data_offsets keys, plus an optional
// __metadata__ entry). No third-party dependencies.

#pragma once

#include "vk_ir.hpp"

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ov::core {
namespace vulkan {
namespace cross_platform {
namespace st_r {

struct st_tensor {
    std::vector<size_t> shape;
    std::vector<float> data;  // f32, row-major
};

// Reads a .safetensors file and returns every tensor converted to f32.
// Throws std::runtime_error on malformed input or unsupported dtype.
std::map<std::string, st_tensor> load_safetensors(const std::string& path);

}  // namespace st_r
}  // namespace cross_platform
}  // namespace vulkan
}  // namespace ov
