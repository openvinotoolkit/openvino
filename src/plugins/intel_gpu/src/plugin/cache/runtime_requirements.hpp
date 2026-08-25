// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>

namespace cldnn {
struct device;
}

namespace ov::intel_gpu::cache {

// On-disk contract for the runtime-requirements block in a compiled GPU model.
inline constexpr uint64_t runtime_requirements_magic = 0x4F5645505F525251ULL;  // "OVEP_RRQ"
inline constexpr uint32_t runtime_requirements_version = 3;

std::string build_runtime_requirements(const cldnn::device& device);
bool is_runtime_requirements_compatible(const std::string& requirements, const cldnn::device& device);

}  // namespace ov::intel_gpu::cache
