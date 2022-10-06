// Copyright (C) 2018-2022 Intel Corporation
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

template <class T>
void shape_infer(const ov::op::v1::SpaceToBatch* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    using ValType = typename std::iterator_traits<typename T::iterator>::value_type::value_type;
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4 && output_shapes.size() == 1);

    const auto& data_shape = input_shapes[0];
    const auto& block_shape = input_shapes[1];
    const auto& pads_begin_shape = input_shapes[2];
    const auto& pads_end_shape = input_shapes[3];
    const ov::Rank data_rank = data_shape.rank();
    bool got_const_data = false;

    auto inputs_same_ps = pads_begin_shape;
    NODE_VALIDATION_CHECK(op,
                          T::merge_into(inputs_same_ps, pads_end_shape) && T::merge_into(inputs_same_ps, block_shape),
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

    if (data_rank.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              (data_shape.size() >= 2),
                              "The data tensor with rank lower than 2 is not supported (data rank: ",
                              data_shape.size(),
                              ")");

        std::vector<int64_t> block_val, pads_begin_val, pads_end_val;

        auto& output_shape = output_shapes[0];
        output_shape.resize(data_shape.size());
        if (get_data_as_int64<T>(1, op, block_val, constant_data) &&
            get_data_as_int64<T>(2, op, pads_begin_val, constant_data) &&
            get_data_as_int64<T>(3, op, pads_end_val, constant_data)) {
            got_const_data = true;
            int64_t block_prod =
                std::accumulate(begin(block_val), end(block_val), int64_t(1), std::multiplies<int64_t>());

            output_shape[0] = data_shape[0] * static_cast<ValType>(block_prod);

            for (size_t idx = 1; idx < output_shape.size(); ++idx) {
                NODE_VALIDATION_CHECK(op, block_val[idx] > 0, "block_shape values must be greater than 0");
                if (data_shape[idx].is_dynamic() && data_shape[idx] == ov::Dimension::dynamic()) {
                    output_shape[idx] = ov::Dimension::dynamic();
                } else {
                    const auto divided =
                        data_shape[idx] + static_cast<ValType>((pads_begin_val[idx] + pads_end_val[idx]));
                    const auto divisor = static_cast<ValType>(block_val[idx]);
                    output_shape[idx] = divided / divisor;
                    check_divided_result(op, output_shape[idx], divided, divisor);
                }
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