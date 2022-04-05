// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/scatter_elements_update.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace v3 {

template <class T>
void shape_infer(const ScatterElementsUpdate* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4 && output_shapes.size() == 1);

    const auto& data_shape = input_shapes[0];
    const auto& indices_shape = input_shapes[1];
    const auto& updates_shape = input_shapes[2];
    const auto& axis_shape = input_shapes[3];
    auto& output_shape = output_shapes[0];
    output_shape = data_shape;

    NODE_VALIDATION_CHECK(op,
                          axis_shape.compatible(T{}) || axis_shape.compatible(T{1}),
                          "Axis input shape are required to be scalar or 1D tensor. ",
                          "Got: ",
                          axis_shape);

    NODE_VALIDATION_CHECK(op,
                          indices_shape.rank().compatible(data_shape.rank()),
                          "Indices rank and data rank are required to be equal. ",
                          "Got: ",
                          indices_shape.rank(),
                          " and: ",
                          data_shape.rank());

    NODE_VALIDATION_CHECK(op,
                          indices_shape.compatible(updates_shape),
                          "Indices and updates input shapes are required to be equal. ",
                          "Got: ",
                          indices_shape,
                          " and: ",
                          updates_shape);

    if (data_shape.rank().is_dynamic())
        return;

    std::vector<int64_t> axis_input;
    if (get_data_as_int64<T>(3, op, axis_input, constant_data)) {
        auto axis = axis_input[0];

        int64_t data_rank_length = data_shape.rank().get_length();
        NODE_VALIDATION_CHECK(op,
                              (-data_rank_length <= axis) && (axis <= data_rank_length - 1),
                              "Axis value has to be in range [-r, r-1] where r is rank of data shape. ",
                              " Data rank: ",
                              data_rank_length,
                              ", range:[",
                              -data_rank_length,
                              ", ",
                              data_rank_length - 1,
                              "]. Got axis value: ",
                              axis);
    }
}

}  // namespace v3
}  // namespace op
}  // namespace ov
