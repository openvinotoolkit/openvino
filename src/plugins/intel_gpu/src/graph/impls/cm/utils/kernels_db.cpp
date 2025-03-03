// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels_db.hpp"
#include <assert.h>
#include <string_view>
#include <array>
#include <utility>
#include "openvino/core/except.hpp"


namespace ov::intel_gpu::cm {

std::string_view CMSourcesDB::get_kernel_template(std::string_view template_name) {
    static std::array sources = {
        #include "gpu_cm_kernel_sources.inc"
    };
    for (const auto& s : sources) {
        if (std::get<0>(s) == template_name) {
            return std::get<1>(s);
        }
    }
    OPENVINO_THROW("Kernel template ", template_name, " not found");
}

std::string_view CMSourcesDB::get_kernel_header(std::string_view header_name) {
    static std::array headers = {
        #include "gpu_cm_kernel_headers.inc"
    };
    for (const auto& s : headers) {
        if (std::get<0>(s) == header_name) {
            return std::get<1>(s);
        }
    }
    OPENVINO_THROW("Kernel header ", header_name, " not found");
}

}  // namespace ov::intel_gpu::cm
