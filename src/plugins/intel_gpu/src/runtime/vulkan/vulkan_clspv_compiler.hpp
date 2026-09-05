// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cldnn::vulkan {

struct vulkan_clspv_compilation final {
    std::vector<uint8_t> spirv;
    std::string diagnostics;
};

/// Vulkan-owned adapter for compiling a materialized OpenCL C translation unit.
class vulkan_clspv_compiler final {
public:
    static std::string identity();
    static std::string canonical_options(const std::string& source_options);

    vulkan_clspv_compilation compile(const std::string& source, const std::string& source_options, const std::string& entry_point) const;
};

}  // namespace cldnn::vulkan
