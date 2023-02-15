// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <openvino/core/validation_util.hpp>
#include <openvino/op/space_to_batch.hpp>
#include <openvino/opsets/opset2.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {

template <class TShape>
std::vector<TShape> shape_infer(const SpaceToBatch* op,
                                const std::vector<TShape>& input_shapes,
                                const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    using TVal = typename TShape::value_type::value_type;
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4);

    const auto& data_shape = input_shapes[0];
    const auto& block_shape = input_shapes[1];
    const auto& pads_begin_shape = input_shapes[2];
    const auto& pads_end_shape = input_shapes[3];

    auto inputs_same_ps = pads_begin_shape;
    NODE_VALIDATION_CHECK(
        op,
        TShape::merge_into(inputs_same_ps, pads_end_shape) && TShape::merge_into(inputs_same_ps, block_shape),
        "block_shape, pads_begin and pads_end inputs must have the same shape. Got: ",
        block_shape,
        ", ",
        pads_begin_shape,
        " and ",
        pads_end_shape);

    NODE_VALIDATION_CHECK(op,
                          inputs_same_ps.rank().compatible(1),
                          "block_shape and pads inputs must have rank 1. Got: ",
                          inputs_same_ps.rank());

    if (data_shape.rank().is_static()) {
        constexpr size_t spatial_dim_offset = 1;
        NODE_VALIDATION_CHECK(op,
                              (data_shape.size() > spatial_dim_offset),
                              "The data tensor with rank lower than 2 is not supported (data rank: ",
                              data_shape.size(),
                              ")");

        auto out_shape = data_shape;
        std::vector<int64_t> block, pads_begin, pads_end;
        if (get_data_as_int64<TShape>(1, op, block, constant_data) &&
            get_data_as_int64<TShape>(2, op, pads_begin, constant_data) &&
            get_data_as_int64<TShape>(3, op, pads_end, constant_data)) {
            TVal block_prod = std::accumulate(begin(block), end(block), 1, std::multiplies<int64_t>());

            out_shape[0] *= block_prod;
            for (auto idx = spatial_dim_offset; idx < out_shape.size(); ++idx) {
                NODE_VALIDATION_CHECK(op, block[idx] > 0, "block_shape values must be greater than 0");
                if (out_shape[idx].is_static() || out_shape[idx] != Dimension::dynamic()) {
                    const auto padded_dim = out_shape[idx] + static_cast<TVal>(pads_begin[idx] + pads_end[idx]);
                    const auto divisor = static_cast<TVal>(block[idx]);
                    out_shape[idx] = padded_dim / divisor;
                    check_divided_result(op, out_shape[idx], padded_dim, divisor);
                }
            }
        }
        return {out_shape};
    } else {
        return {PartialShape::dynamic()};
    }
}

template <class TShape>
void shape_infer(const SpaceToBatch* op,
                 const std::vector<TShape>& input_shapes,
                 std::vector<TShape>& output_shapes,
                 const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    output_shapes = shape_infer(op, input_shapes, constant_data);
}

}  // namespace v1
}  // namespace op
}  // namespace ov
