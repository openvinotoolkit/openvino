// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/strided_slice.hpp"

#include <stdio.h>

#include <cmath>

#include "openvino/core/memory_util.hpp"
#include "openvino/reference/reshape.hpp"
#include "openvino/reference/reverse.hpp"
#include "openvino/reference/slice.hpp"
#include "openvino/runtime/aligned_buffer.hpp"

namespace ov {
namespace reference {

void strided_slice(const char* arg,
                   char* out,
                   const Shape& arg_shape,
                   const op::util::SlicePlan& sp,
                   const element::Type& elem_type) {
    auto hasZeroDims = [](const Shape& shape) -> bool {
        return std::any_of(shape.begin(), shape.end(), [](const size_t& dim) {
            return dim == 0;
        });
    };
    if (hasZeroDims(sp.reshape_in_shape) || hasZeroDims(sp.reshape_out_shape)) {
        return;
    }
    OPENVINO_ASSERT(elem_type.bitwidth() >= 8,
                    "StridedSlice reference implementation does not support element types with bitwidth less than 8.");

    auto in_memory_size = ov::util::get_memory_size_safe(elem_type, sp.reshape_in_shape).value_or(1);
    ov::AlignedBuffer slice_out_buffer(in_memory_size);
    slice(arg,
          slice_out_buffer.get_ptr<char>(),
          arg_shape,
          Coordinate(sp.begins.begin(), sp.begins.end()),
          Coordinate(sp.ends.begin(), sp.ends.end()),
          Strides(sp.strides.begin(), sp.strides.end()),
          sp.reshape_in_shape,
          elem_type.size());

    auto out_memory_size = ov::util::get_memory_size_safe(elem_type, sp.reshape_out_shape).value_or(1);
    ov::AlignedBuffer reshape_out_buffer(out_memory_size);

    auto copy_size = std::min(in_memory_size, out_memory_size);

    reshape(slice_out_buffer.get_ptr<char>(), reshape_out_buffer.get_ptr<char>(), copy_size);

    reverse(reshape_out_buffer.get_ptr<char>(),
            out,
            sp.reshape_out_shape,
            sp.reshape_out_shape,
            sp.reverse_axes,
            elem_type.size());
}
}  // namespace reference
}  // namespace ov
