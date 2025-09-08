// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ov_ops/glu.hpp"
#include "utils.hpp"
#include "variadic_split_shape_inference.hpp"

namespace ov {
namespace op {
namespace internal {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const GLU* op, const std::vector<TShape>& input_shapes) {
    const auto inputs_count = input_shapes.size();
    NODE_SHAPE_INFER_CHECK(op, input_shapes, inputs_count == 1);

    int64_t axis = op->get_axis();
    std::vector<int64_t> split_lengths = {op->get_split_lengths(), -1};
    std::unordered_map<size_t, ov::Tensor> const_data;
    const_data.emplace(1, ov::Tensor(ov::element::i64, ov::Shape{}, &axis));
    const_data.emplace(2, ov::Tensor(ov::element::i64, ov::Shape{split_lengths.size()}, split_lengths.data()));

    const ov::Shape split_len_size{split_lengths.size()};
    const ov::Shape scalar{};
    std::vector<TShape> variadic_split_input_shapes{input_shapes[0], scalar, split_len_size};

    return {std::move(
        ov::op::variadic_split::shape_infer(op, variadic_split_input_shapes, ov::make_tensor_accessor(const_data))[0])};
}
}  // namespace internal
}  // namespace op
}  // namespace ov
