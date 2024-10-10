// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/search_sorted.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v15 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const SearchSorted* op, const std::vector<TShape>& input_shapes) {
    // [HACK]: By convention, shape_infer should also perform node validation..
    op->validate();
    const auto& sorted_shape = input_shapes[0];
    const auto& values_shape = input_shapes[1];
    auto output_shape = values_shape;
    TShape::merge_into(output_shape, sorted_shape);

    if (output_shape.rank().is_static()) {
        auto last_it = output_shape.end() - 1;
        if (values_shape.rank().is_static()) {
            *last_it = *(input_shapes[1].end() - 1);
        } else {
            *last_it = Dimension::dynamic();
        }
    }

    return {std::move(output_shape)};
}
}  // namespace v15
}  // namespace op
}  // namespace ov
