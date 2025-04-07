// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "dimension_util.hpp"
#include "openvino/op/batch_to_space.hpp"
#include "openvino/opsets/opset2.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const BatchToSpace* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    using namespace ov::util;
    using ValType = typename TShape::value_type::value_type;
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4);

    const auto& data_shape = input_shapes[0];
    const auto& block_shape = input_shapes[1];
    const auto& crops_begin_shape = input_shapes[2];
    const auto& crops_end_shape = input_shapes[3];

    TRShape inputs_same_ps = crops_begin_shape;
    NODE_VALIDATION_CHECK(
        op,
        TRShape::merge_into(inputs_same_ps, crops_end_shape) && TRShape::merge_into(inputs_same_ps, block_shape),
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

    auto output_shapes = std::vector<TRShape>(1);
    auto& out_shape = output_shapes[0];

    const auto data_rank = data_shape.rank();
    if (data_rank.is_static()) {
        constexpr size_t spatial_dim_offset = 1;
        const auto data_rank_size = data_shape.size();

        NODE_VALIDATION_CHECK(op,
                              (data_rank_size > spatial_dim_offset),
                              "data input must have rank greater or equal than 2. Got: ",
                              data_rank_size);
        if (inputs_same_ps.is_static()) {
            NODE_VALIDATION_CHECK(
                op,
                static_cast<int64_t>(data_rank.get_length()) == static_cast<int64_t>(inputs_same_ps[0].get_length()),
                "block_shape and crop inputs must have same number of elements "
                "as data input rank. Got: ",
                inputs_same_ps[0],
                " and ",
                data_rank);
        }

        out_shape.reserve(data_rank_size);

        const auto blocks = get_input_const_data_as<TRShape, int64_t>(op, 1, tensor_accessor);
        if (blocks) {
            NODE_VALIDATION_CHECK(op,
                                  std::none_of(begin(*blocks), end(*blocks), cmp::Less<int64_t>(1)),
                                  "Elements of block_shape input must be greater or equal to one.");
            const auto divisor = static_cast<ValType>(
                std::accumulate(begin(*blocks), end(*blocks), int64_t(1), std::multiplies<int64_t>()));
            out_shape.push_back(data_shape[0] / divisor);
            check_divided_result(op, out_shape[0], data_shape[0], divisor);
        } else {
            out_shape.emplace_back(dim::inf_bound);
        }

        const auto crops_begin = get_input_const_data_as<TRShape, int64_t>(op, 2, tensor_accessor);
        const auto crops_end = get_input_const_data_as<TRShape, int64_t>(op, 3, tensor_accessor);
        if (crops_begin && crops_end) {
            auto& crops_begin_val = *crops_begin;
            auto& crops_end_val = *crops_end;

            constexpr auto is_invalid_crop = cmp::Less<int64_t>(0);
            NODE_VALIDATION_CHECK(op,
                                  std::none_of(begin(crops_begin_val), end(crops_begin_val), is_invalid_crop) &&
                                      std::none_of(begin(crops_end_val), end(crops_end_val), is_invalid_crop),
                                  "Elements of crops_begin and crops_end inputs must be greater or equal to zero.");

            if (blocks) {
                for (auto idx = spatial_dim_offset; idx < data_rank_size; ++idx) {
                    auto d = data_shape[idx] * static_cast<ValType>((*blocks)[idx]);
                    auto crop = static_cast<ValType>(crops_begin_val[idx] + crops_end_val[idx]);
                    NODE_VALIDATION_CHECK(
                        op,
                        d.is_dynamic() || crop <= d.get_length(),
                        "crops_begin[i] + crops_end[i] must be less or equal to block_shape[i] * input_shape[i]");

                    out_shape.push_back(d - crop);
                }
            } else {
                const auto block = Dimension(1, dim::inf_bound);
                for (auto idx = spatial_dim_offset; idx < data_rank_size; ++idx) {
                    auto d = data_shape[idx] * block;
                    auto crop = static_cast<ValType>(crops_begin_val[idx] + crops_end_val[idx]);
                    out_shape.push_back(d - crop);
                }
            }
        } else {
            out_shape.insert(out_shape.end(), data_rank_size - spatial_dim_offset, Dimension::dynamic());
        }
    } else {
        out_shape = PartialShape::dynamic();
    }

    return output_shapes;
}
}  // namespace v1
}  // namespace op
}  // namespace ov
