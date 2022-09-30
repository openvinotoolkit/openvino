// Copyright (C) 2018-2022 Intel Corporation
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

template <class T>
void shape_infer(const ov::op::v1::BatchToSpace* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    using ValType = typename std::iterator_traits<typename T::iterator>::value_type::value_type;
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4 && output_shapes.size() == 1);
    const auto& data_shape = input_shapes[0];
    const auto& block_shape = input_shapes[1];
    const auto& crops_begin_shape = input_shapes[2];
    const auto& crops_end_shape = input_shapes[3];
    bool got_const_data = false;

    auto inputs_same_ps = crops_begin_shape;
    NODE_VALIDATION_CHECK(op,
                          T::merge_into(inputs_same_ps, crops_end_shape) && T::merge_into(inputs_same_ps, block_shape),
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
        NODE_VALIDATION_CHECK(op,
                              (data_rank.get_length() >= 2),
                              "data input must have rank greater or equal than 2. Got: ",
                              data_rank.get_length());
        if (inputs_same_ps.is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  data_rank.get_length() == inputs_same_ps[0].get_length(),
                                  "block_shape and crop inputs must have same number of elements "
                                  "as data input rank. Got: ",
                                  inputs_same_ps[0],
                                  " and ",
                                  data_rank);
        }

        auto& output_shape = output_shapes[0];
        output_shape.resize(data_shape.size());

        std::vector<int64_t> block_val, crops_begin_val, crops_end_val;

        if (get_data_as_int64<T>(1, op, block_val, constant_data) &&
            get_data_as_int64<T>(2, op, crops_begin_val, constant_data) &&
            get_data_as_int64<T>(3, op, crops_end_val, constant_data)) {
            got_const_data = true;
            bool block_vals_valid = std::all_of(begin(block_val), end(block_val), [](int64_t elem) {
                return elem >= 1;
            });
            NODE_VALIDATION_CHECK(op,
                                  block_vals_valid,
                                  "Elements of block_shape input must be greater or equal to one.");

            bool crops_begin_vals_valid = std::all_of(begin(crops_begin_val), end(crops_begin_val), [](int64_t elem) {
                return elem >= 0;
            });
            bool crops_end_vals_valid = std::all_of(begin(crops_end_val), end(crops_end_val), [](int64_t elem) {
                return elem >= 0;
            });
            NODE_VALIDATION_CHECK(op,
                                  crops_begin_vals_valid && crops_end_vals_valid,
                                  "Elements of crops_begin and crops_end inputs must be greater or equal to zero.");
            if (data_shape.is_static()) {
                for (size_t idx = 0; idx < data_shape.size(); idx++) {
                    const bool is_valid_crops_and_shape =
                        crops_begin_val[idx] + crops_end_val[idx] <= block_val[idx] * data_shape[idx].get_length();
                    NODE_VALIDATION_CHECK(op,
                                          is_valid_crops_and_shape,
                                          "crops_begin[i] + crops_end[i] must be less or equal to "
                                          "block_shape[i] * input_shape[i]");
                }
            }

            int64_t block_prod =
                std::accumulate(begin(block_val), end(block_val), int64_t(1), std::multiplies<int64_t>());
            const auto divisor = static_cast<ValType>(block_prod);

            output_shape[0] = data_shape[0] / divisor;
            check_divided_result(op, output_shape[0], data_shape[0], divisor);

            for (size_t idx = 1; idx < data_shape.size(); idx++) {
                output_shape[idx] = data_shape[idx] * static_cast<ValType>(block_val[idx]) -
                                    static_cast<ValType>(crops_begin_val[idx]) -
                                    static_cast<ValType>(crops_end_val[idx]);
            }
        }
    }
    if (!got_const_data)
        // For PartialShape, Set the output to be dynamic;
        // For StaticShape, throw error caused by implicitly constructing StaticShape with PartialShape argument;
        output_shapes[0] = ov::PartialShape::dynamic(data_rank);
}

}  // namespace v1
}  // namespace op
}  // namespace ov