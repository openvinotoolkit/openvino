// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"

namespace ov::intel_gpu::cm {

enum class MemLayout { row_major, col_major };

inline std::string get_xetla_mem_layout(MemLayout layout) {
    switch (layout) {
    case MemLayout::row_major:
        return "mem_layout::row_major";
    case MemLayout::col_major:
        return "mem_layout::col_major";
    default:
        OPENVINO_THROW("Unsupported XeTLA memory layout!");
    }
}

inline std::string ov_to_xetla_dtype(ov::element::Type type) {
    switch (type) {
    case ov::element::Type_t::f16:
        return "fp16";
    case ov::element::Type_t::bf16:
        return "bf16";
    case ov::element::Type_t::f32:
        return "float";
    default:
        OPENVINO_THROW("Unsupported XeTLA data type!");
    }
}
}  // namespace ov::intel_gpu::cm
