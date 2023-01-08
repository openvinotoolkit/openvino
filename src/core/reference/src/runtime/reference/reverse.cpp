// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/reverse.hpp"

#include <cmath>
#include <cstring>
#include <iterator>

#include "ngraph/check.hpp"
#include "ngraph/coordinate_range.hpp"

using namespace ngraph;

namespace ngraph {
namespace runtime {
namespace reference {
void reverse(const char* arg,
             char* out,
             const Shape& arg_shape,
             const Shape& out_shape,
             const AxisSet& reversed_axes,
             size_t elem_size) {
    NGRAPH_CHECK(shape_size(arg_shape) == shape_size(out_shape));

    const bool nothing_to_revers = reversed_axes.empty();
    if (nothing_to_revers) {
        std::memcpy(out, arg, shape_size(arg_shape) * elem_size);
        return;
    }

    auto dst_mem = out;
    for (auto range : coordinates::reverse(arg_shape, reversed_axes)) {
        auto src_index = range.begin_index;

        if (range.direction == coordinates::Direction::forward) {
            for (size_t i = 0; i < range.element_number; src_index += range.step, ++i) {
                const auto src_mem = arg + src_index * elem_size;
                std::memcpy(dst_mem, src_mem, elem_size);
                std::advance(dst_mem, elem_size);
            }
        } else {
            for (size_t i = 0; i < range.element_number; src_index -= range.step, ++i) {
                const auto src_mem = arg + src_index * elem_size;
                std::memcpy(dst_mem, src_mem, elem_size);
                std::advance(dst_mem, elem_size);
            }
        }
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
