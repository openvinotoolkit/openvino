// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/strided_slice.hpp"

#include <stdio.h>

#include <cmath>

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
                   size_t elem_type) {
    auto hasZeroDims = [](const Shape& shape) -> bool {
        return std::any_of(shape.begin(), shape.end(), [](const size_t& dim) {
            return dim == 0;
        });
    };
    if (hasZeroDims(sp.reshape_in_shape) || hasZeroDims(sp.reshape_out_shape)) {
        return;
    }

    ov::AlignedBuffer slice_out_buffer(shape_size(sp.reshape_in_shape) * elem_type);
    slice(arg,
          slice_out_buffer.get_ptr<char>(),
          arg_shape,
          Coordinate(sp.begins.begin(), sp.begins.end()),
          Coordinate(sp.ends.begin(), sp.ends.end()),
          Strides(sp.strides.begin(), sp.strides.end()),
          sp.reshape_in_shape,
          elem_type);

    ov::AlignedBuffer reshape_out_buffer(shape_size(sp.reshape_out_shape) * elem_type);
    reshape(slice_out_buffer.get_ptr<char>(), reshape_out_buffer.get_ptr<char>(), sp.reshape_in_shape, elem_type);

    reverse(reshape_out_buffer.get_ptr<char>(),
            out,
            sp.reshape_out_shape,
            sp.reshape_out_shape,
            sp.reverse_axes,
            elem_type);
}
}  // namespace reference
}  // namespace ov
