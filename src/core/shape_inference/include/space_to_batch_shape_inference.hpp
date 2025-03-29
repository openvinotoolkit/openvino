// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "dimension_util.hpp"
#include "openvino/op/space_to_batch.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const SpaceToBatch* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    using namespace ov::util;
    using TVal = typename TShape::value_type::value_type;
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4);

    const auto& data_shape = input_shapes[0];
    const auto& block_shape = input_shapes[1];
    const auto& pads_begin_shape = input_shapes[2];
    const auto& pads_end_shape = input_shapes[3];

    auto inputs_same_ps = static_cast<TRShape>(pads_begin_shape);
    NODE_VALIDATION_CHECK(
        op,
        TRShape::merge_into(inputs_same_ps, pads_end_shape) && TRShape::merge_into(inputs_same_ps, block_shape),
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

    auto output_shapes = std::vector<TRShape>{data_shape};
    if (data_shape.rank().is_static()) {
        constexpr size_t spatial_dim_offset = 1;
        const auto data_rank_size = data_shape.size();
        NODE_VALIDATION_CHECK(op,
                              (data_rank_size > spatial_dim_offset),
                              "The data tensor with rank lower than 2 is not supported (data rank: ",
                              data_rank_size,
                              ")");

        auto& out_shape = output_shapes[0];
        out_shape.resize(0);

        auto blocks = get_input_const_data_as<TShape, int64_t>(op, 1, ta);
        if (blocks) {
            TVal block_prod = std::accumulate(begin(*blocks), end(*blocks), int64_t(1), std::multiplies<int64_t>());
            out_shape.push_back(data_shape[0] * block_prod);
        } else {
            out_shape.emplace_back(dim::inf_bound);
        }

        auto pads_begin = get_input_const_data_as<TShape, int64_t>(op, 2, ta);
        auto pads_end = get_input_const_data_as<TShape, int64_t>(op, 3, ta);
        if (blocks && pads_begin && pads_end) {
            for (auto idx = spatial_dim_offset; idx < data_rank_size; ++idx) {
                NODE_VALIDATION_CHECK(op, (*blocks)[idx] > 0, "block_shape values must be greater than 0");

                const auto padded_dim = data_shape[idx] + static_cast<TVal>((*pads_begin)[idx] + (*pads_end)[idx]);
                const auto divisor = static_cast<TVal>((*blocks)[idx]);

                if (static_cast<int64_t>(padded_dim.get_max_length()) == dim::inf_bound) {
                    out_shape.emplace_back(ceil_div(padded_dim.get_min_length(), divisor), dim::inf_bound);
                } else {
                    out_shape.push_back(padded_dim / divisor);
                }

                check_divided_result(op, out_shape[idx], padded_dim, divisor);
            }
        } else {
            out_shape.insert(out_shape.end(), data_rank_size - spatial_dim_offset, dim::inf_bound);
        }
    }

    return output_shapes;
}
}  // namespace v1
}  // namespace op
}  // namespace ov
