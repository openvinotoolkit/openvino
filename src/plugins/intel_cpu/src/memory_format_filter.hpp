// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

namespace ov::intel_cpu {

struct MemoryFormatFilter {
    std::vector<dnnl::memory::format_tag> input;
    std::vector<dnnl::memory::format_tag> output;

    [[nodiscard]] bool empty() const {
        return input.empty() && output.empty();
    }
};

}  // namespace ov::intel_cpu
