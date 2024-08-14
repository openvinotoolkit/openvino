// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "openvino/op/string_tensor_unpack.hpp"
#include "shape_infer_type_utils.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v15 {
namespace util {
static inline Tensor get_string_tensor(const Node* op, const ITensorAccessor& tensor_accessor) {
    if (auto t = tensor_accessor(0)) {
        return t;
    } else if (const auto& constant = as_type_ptr<v0::Constant>(op->get_input_node_shared_ptr(0))) {
        return constant->get_tensor_view();
    } else {
        return {};
    }
}
}  // namespace util
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const StringTensorUnpack* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1);
    const auto& data_shape = input_shapes[0];
    auto output_shapes = std::vector<TRShape>{data_shape, data_shape};
    if (const auto string_data = util::get_string_tensor(op, tensor_accessor)) {
        const auto string_count = string_data.get_size();
        const auto tensor_data = string_data.data<std::string>();
        size_t total_length = 0;
        for (auto it = tensor_data; it != std::next(tensor_data, string_count); ++it) {
            total_length += (*it).length();
        }
        output_shapes.emplace_back(TRShape{static_cast<typename TRShape::value_type>(total_length)});
    } else {
        output_shapes.emplace_back(ov::PartialShape{ov::Dimension::dynamic()});
    }

    return output_shapes;
}
}  // namespace v15
}  // namespace op
}  // namespace ov
