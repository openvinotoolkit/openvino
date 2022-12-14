// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/tile.hpp>

#include "utils.hpp"
namespace ov {
namespace op {
namespace v0 {

template <class T>
void shape_infer(const Tile* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;

    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);

    const auto& repeats_shape = input_shapes[1];
    NODE_VALIDATION_CHECK(op, repeats_shape.rank().compatible(1), "Tile repeats must be of rank 1");

    const auto& arg_shape = input_shapes[0];
    auto& output_shape = output_shapes[0];
    std::vector<int64_t> axes_val;

    // Get repeats
    bool axes_are_known = get_data_as_int64<T>(1, op, axes_val, constant_data);
    const auto& arg_rank = arg_shape.rank();
    if (arg_rank.is_static() && (axes_are_known || repeats_shape[0].is_static())) {
        // try to specify rank
        int64_t data_rank = arg_shape.size();
        int64_t repeats_rank = axes_are_known ? axes_val.size() : repeats_shape[0].get_length();
        auto output_rank = std::max(data_rank, repeats_rank);
        output_shape.resize(output_rank);
        // if have constant axes, compute new axes
        if (axes_are_known) {
            auto remain_arg = output_rank - data_rank;
            auto remain_axes = output_rank - repeats_rank;
            for (size_t i = 0; i < static_cast<size_t>(output_rank); i++) {
                auto data_tmp = i < static_cast<size_t>(remain_arg) ? DimType(1) : arg_shape[i - (remain_arg)];
                int64_t repeat_tmp =
                    i < static_cast<size_t>(remain_axes) ? 1 : std::max<int64_t>(0, axes_val[i - remain_axes]);
                output_shape[i] = (repeat_tmp > 0) ? data_tmp * repeat_tmp : DimType(0);
            }
        }
    } else {
        // can't deduce shape, set default value
        output_shape = PartialShape::dynamic();
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov
