// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/region_yolo.hpp>

#include "utils.hpp"
namespace ov {
namespace op {
namespace v0 {

template <class T>
void shape_infer(const RegionYolo* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 1) && output_shapes.size() == 1);

    const auto& input_shape = input_shapes[0];
    const auto& input_rank = input_shape.rank();
    auto& output_shape = output_shapes[0];

    NODE_VALIDATION_CHECK(op, input_rank.compatible(4), "Input must be a tensor of rank 4, but got ", input_rank);

    if (input_rank.is_static()) {
        int64_t end_axis = op->m_end_axis;
        if (end_axis < 0) {
            end_axis += static_cast<int>(input_shape.size());
        }

        if (op->m_do_softmax) {
            output_shape.resize(0);
            auto axis = ov::normalize_axis(op, op->m_axis, input_rank);
            DimType flat_dim = 1;
            for (int64_t i = 0; i < axis; i++) {
                output_shape.push_back(input_shape[i]);
            }
            for (int64_t i = axis; i < end_axis + 1; i++) {
                flat_dim *= input_shape[i];
            }
            output_shape.push_back(flat_dim);
            for (size_t i = end_axis + 1; i < input_shape.size(); i++) {
                output_shape.push_back(input_shape[i]);
            }
        } else {
            output_shape = T({input_shape[0],
                              static_cast<typename DimType::value_type>(
                                  (op->get_num_classes() + op->get_num_coords() + 1) * op->get_mask().size()),
                              input_shape[2],
                              input_shape[3]});
        }
    } else {
        output_shape = ov::PartialShape::dynamic(ov::Rank(1, 4));
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov
