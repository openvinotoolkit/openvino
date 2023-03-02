// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <openvino/core/validation_util.hpp>
#include <openvino/op/batch_to_space.hpp>
#include <openvino/opsets/opset2.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {

template <class TShape>
std::vector<TShape> shape_infer(const BatchToSpace* op,
                                const std::vector<TShape>& input_shapes,
                                const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    using ValType = typename TShape::value_type::value_type;
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4);

    const auto& data_shape = input_shapes[0];
    const auto& block_shape = input_shapes[1];
    const auto& crops_begin_shape = input_shapes[2];
    const auto& crops_end_shape = input_shapes[3];

    auto inputs_same_ps = crops_begin_shape;
    NODE_VALIDATION_CHECK(
        op,
        TShape::merge_into(inputs_same_ps, crops_end_shape) && TShape::merge_into(inputs_same_ps, block_shape),
        "block_shape, crops_begin and crops_end inputs must have the same shape. Got: ",
        block_shape,
        ", ",
        crops_begin_shape,
        " and ",
        crops_end_shape);

    NODE_VALIDATION_CHECK(op,
                          inputs_same_ps.rank().compatible(1),
                          "block_shape and crops inputs must have rank 1. Got: ",
                          inputs_same_ps.rank());

    const ov::Rank data_rank = data_shape.rank();
    if (data_rank.is_static()) {
        constexpr size_t spatial_dim_offset = 1;
        NODE_VALIDATION_CHECK(op,
                              (data_shape.size() > spatial_dim_offset),
                              "data input must have rank greater or equal than 2. Got: ",
                              data_shape.size());
        if (inputs_same_ps.is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  data_rank.get_length() == inputs_same_ps[0].get_length(),
                                  "block_shape and crop inputs must have same number of elements "
                                  "as data input rank. Got: ",
                                  inputs_same_ps[0],
                                  " and ",
                                  data_rank);
        }

        auto out_shape = data_shape;
        std::vector<int64_t> block_val, crops_begin_val, crops_end_val;

        if (get_data_as_int64<TShape>(1, op, block_val, constant_data) &&
            get_data_as_int64<TShape>(2, op, crops_begin_val, constant_data) &&
            get_data_as_int64<TShape>(3, op, crops_end_val, constant_data)) {
            NODE_VALIDATION_CHECK(op,
                                  std::none_of(begin(block_val), end(block_val), cmp::Less<int64_t>(1)),
                                  "Elements of block_shape input must be greater or equal to one.");

            constexpr auto is_invalid_crop = cmp::Less<int64_t>(0);
            NODE_VALIDATION_CHECK(op,
                                  std::none_of(begin(crops_begin_val), end(crops_begin_val), is_invalid_crop) &&
                                      std::none_of(begin(crops_end_val), end(crops_end_val), is_invalid_crop),
                                  "Elements of crops_begin and crops_end inputs must be greater or equal to zero.");

            const auto divisor = static_cast<ValType>(
                std::accumulate(begin(block_val), end(block_val), int64_t(1), std::multiplies<int64_t>()));

            out_shape[0] /= divisor;
            check_divided_result(op, out_shape[0], data_shape[0], divisor);

            for (auto idx = spatial_dim_offset; idx < out_shape.size(); ++idx) {
                out_shape[idx] *= static_cast<ValType>(block_val[idx]);
                auto crop = static_cast<ValType>(crops_begin_val[idx] + crops_end_val[idx]);
                NODE_VALIDATION_CHECK(
                    op,
                    out_shape[idx].is_dynamic() || crop <= out_shape[idx].get_length(),
                    "crops_begin[i] + crops_end[i] must be less or equal to block_shape[i] * input_shape[i]");

                out_shape[idx] = out_shape[idx] - crop;
            }
        }
        return {out_shape};
    } else {
        return {PartialShape::dynamic()};
    }
}

template <class TShape>
void shape_infer(const ov::op::v1::BatchToSpace* op,
                 const std::vector<TShape>& input_shapes,
                 std::vector<TShape>& output_shapes,
                 const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    output_shapes = shape_infer(op, input_shapes, constant_data);
}

}  // namespace v1
}  // namespace op
}  // namespace ov
