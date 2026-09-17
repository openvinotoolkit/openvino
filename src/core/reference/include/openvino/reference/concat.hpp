// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
/// \brief Concatenates byte-copyable data (elem_size is the size in bytes of a single element).
void concat(const std::vector<const char*>& args,
            char* out,
            const std::vector<Shape>& in_shapes,
            const Shape& out_shape,
            int64_t concatenation_axis,
            size_t elem_size);

/// \brief Concatenates string data.
void concat(const std::vector<const std::string*>& args,
            std::string* out,
            const std::vector<Shape>& in_shapes,
            const Shape& out_shape,
            int64_t concatenation_axis);

/// \brief Concatenates sub-byte packed data (bitwidth < 8: u1, u2, u4, i4, nf4, f4e2m1, all stored as int8_t).
void concat(const std::vector<const int8_t*>& args,
            int8_t* out,
            const std::vector<Shape>& in_shapes,
            const Shape& out_shape,
            int64_t concatenation_axis,
            size_t bitwidth);

}  // namespace reference
}  // namespace ov
