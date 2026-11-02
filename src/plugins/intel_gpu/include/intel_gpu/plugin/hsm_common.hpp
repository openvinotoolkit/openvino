// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "openvino/runtime/hsm_format.hpp"

namespace ov::intel_gpu::hsm {

namespace wire = ov::runtime::hsm;
using Crypt = std::function<std::string(const std::string&)>;

// GPU-local inline tag 0 describes the compiled graph payload, not a shared weights codec.
inline constexpr auto payload_info_tag = wire::SectionTag::make_device_tag(0, true);
inline constexpr size_t max_requirements_size = 4096;
inline constexpr uint32_t encrypted = 1;

struct PayloadInfo {
    uint32_t version = 1;
    uint32_t cache_mode = 0;  // 0: OPTIMIZE_SPEED; 1: OPTIMIZE_SIZE.
    uint32_t flags = 0;
    uint32_t requirements_version = 2;
    uint64_t reserved = 0;
};
static_assert(sizeof(PayloadInfo) == 24);

}  // namespace ov::intel_gpu::hsm
