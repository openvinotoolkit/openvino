// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/tile.hpp>

#include "shape_infer_utils.hpp"
namespace ov {
namespace op {
namespace v0 {

template <class T>
void shape_infer(const Tile* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);
    const auto& arg_shape = input_shapes[0];
    const auto& repeats_shape = input_shapes[1];
    auto& output_shape = output_shapes[0];
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;

    if (repeats_shape.is_static() && repeats_shape.rank().get_length() > 0 && arg_shape.rank().is_static()) {
        auto data_rank = arg_shape.rank().get_length();
        auto repeats_rank = repeats_shape.rank().get_length();
        auto output_rank = std::max(data_rank, repeats_rank);
        output_shape.resize(output_rank);
        for (size_t i = 0; i < output_rank; i++) {
            auto data_tmp = i < output_rank - data_rank ? DimType(1) : arg_shape[i - (output_rank - data_rank)];
            auto repeat_tmp =
                i < output_rank - repeats_rank ? DimType(1) : repeats_shape[i - (output_rank - repeats_rank)];
            output_shape[i] = data_tmp * repeat_tmp;
        }
    } else {
        ShapeInfer::default_work(output_shape);
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov