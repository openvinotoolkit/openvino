// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Experimental CM kernel template database implementation. See kernels_db.hpp.
//
// The generated `experimental_cm_kernel_sources.inc` (produced at build time by the stock
// kernels_db_gen.py -- see experimental/CMakeLists.txt) contains
//
//     std::make_pair<std::string_view, std::string_view>("<kernel-name>", R"__krnl(...)__krnl"),
//
// entries which we fold into a static array here. Kernel names map exactly to the template_name
// argument the experimental generators pass to their KernelGenerator base (which in turn round-trips
// through `ExperimentalKernelGenerator` to this resolver).

#include "kernels_db.hpp"

#include <array>
#include <string_view>

#include "openvino/core/except.hpp"

namespace ov::intel_gpu::experimental::cm {

std::string_view ExperimentalSourcesDB::get_kernel_template(std::string_view template_name) {
    static std::array sources = {
#include "experimental_cm_kernel_sources.inc"
    };
    for (const auto& s : sources) {
        if (std::get<0>(s) == template_name) {
            return std::get<1>(s);
        }
    }
    OPENVINO_THROW("Experimental CM Kernel template ", template_name, " not found");
}

}  // namespace ov::intel_gpu::experimental::cm