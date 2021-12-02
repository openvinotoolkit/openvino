// Copyright (C) 2018-2021 Intel Corporation
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
void shape_infer(const RegionYolo* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 1) && output_shapes.size() == 1);

    const auto& input_shape = input_shapes[0];
    auto& output_shape = output_shapes[0];
    if (input_shape.rank().is_static()) {
        int end_axis = op->get_end_axis();
        if (end_axis < 0) {
            end_axis += input_shape.size();
        }

        if (op->get_do_softmax()) {
            output_shape.resize(0);
            auto axis = op->get_axis();
            size_t flat_dim = 1;
            for (int64_t i = 0; i < axis; i++) {
                output_shape.push_back(input_shape[i]);
            }
            for (int64_t i = axis; i < end_axis + 1; i++) {
                if (input_shape[i].is_dynamic()) {
                    flat_dim = -1;
                    break;
                }
                flat_dim *= input_shape[i].get_length();
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
        output_shape = ov::PartialShape::dynamic();
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov
