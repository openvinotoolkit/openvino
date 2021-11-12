// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/tile.hpp>

#include "shape_infer_utils.hpp"
#include "utils.hpp"
namespace ov {
namespace op {
namespace v0 {

template <class T>
void shape_infer(const Tile* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);
    const auto& arg_shape = input_shapes[0];
    auto& repeats_shape = input_shapes[1];
    auto& output_shape = output_shapes[0];
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    std::vector<int64_t> axes_val;
    NODE_VALIDATION_CHECK(op, repeats_shape.rank().compatible(1), "PartialShape of repeats must be of rank 1");
    //Get repeats
    bool axes_are_known = get_data_as_int64<T>(1, op, axes_val, constant_data);
    if (axes_are_known && arg_shape.rank().is_static()) {
        auto data_rank = arg_shape.size();
        auto repeats_rank = axes_val.size();
        auto output_rank = std::max(data_rank, repeats_rank);
        output_shape.resize(output_rank);
        for (size_t i = 0; i < output_rank; i++) {
            auto data_tmp = i < output_rank - data_rank ? DimType(1) : arg_shape[i - (output_rank - data_rank)];
            auto repeat_tmp =
                i < output_rank - repeats_rank ? DimType(1) : axes_val[i - (output_rank - repeats_rank)];
            output_shape[i] = data_tmp * repeat_tmp;
        }
    } else {
        ShapeInfer::default_work(output_shape);
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov