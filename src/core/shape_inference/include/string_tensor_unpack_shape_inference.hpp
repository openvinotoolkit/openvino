// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "openvino/op/string_tensor_unpack.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v15 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const StringTensorUnpack* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1);
    const auto& data_shape = input_shapes[0];
    auto output_shapes = std::vector<TRShape>(3);
    
    if (data_shape.is_static()) {
        const uint64_t string_count = data_shape[0].get_length();

        // output 1 and 2: begins and ends
        output_shapes[0] = ov::Shape{string_count};
        output_shapes[1] = ov::Shape{string_count};

        // output 3: symbols
        const auto strings = ov::op::get_input_const_data_as<TRShape, uint8_t>(op, 0, tensor_accessor);
        size_t total_length = 0;
        for(size_t i = 0; i <= string_count; ++i)
            total_length += (*strings)[i];
        output_shapes[2] = ov::Shape{total_length};
    }
    return output_shapes;
}
}  // namespace v15
}  // namespace op
}  // namespace ov
