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

    // NOTE: The output shape is the same as the values shape - always.
    // The code below only tries to infer dynamic dims from sorted_shape - if possible.

    // 1. If we know that the sorted sequence is 1D, than output shape can be anything.
    if (sorted_shape.rank().is_static() && sorted_shape.rank().get_length() == 1) {
        return {std::move(output_shape)};
    }

    // 2. ND tensor case or rank not known.
    auto sorted_shape_last_dynamic = sorted_shape;
    if (sorted_shape.rank().is_static()) {
        sorted_shape_last_dynamic[sorted_shape.rank().get_length() - 1] = Dimension::dynamic();
    }

    const bool sorted_values_merge_success = TShape::merge_into(output_shape, sorted_shape_last_dynamic);

    NODE_VALIDATION_CHECK(op, sorted_values_merge_success, "Shapes of sorted sequence and values are not compatible.");

    return {std::move(output_shape)};
}
}  // namespace v15
}  // namespace op
}  // namespace ov
