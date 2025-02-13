// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string_view>
#include <cctype>

namespace ov::intel_gpu::ocl {
struct OCLSourcesDB {
    static std::string_view get_kernel_template(std::string_view template_name);
    static std::string_view get_kernel_header(std::string_view header_name);
};

}  // namespace ov::intel_gpu::ocl
