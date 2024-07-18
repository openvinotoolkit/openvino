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
static Tensor get_string_tensor(const Node* op, size_t idx, const ITensorAccessor& tensor_accessor) {
    if (auto t = tensor_accessor(idx)) {
        return t;
    } else {
        const auto& constant = as_type_ptr<opset1::Constant>(op->get_input_node_shared_ptr(idx));
        return constant->get_tensor_view();
    }
}
template <class TRShape>
static void handle_else_case(std::vector<TRShape>& output_shapes, const ov::op::v15::StringTensorUnpack* op) {
    if (!std::is_same<TRShape, PartialShape>::value) {
        const auto& constant = as_type_ptr<opset1::Constant>(op->get_input_node_shared_ptr(0));
        NODE_VALIDATION_CHECK(op, constant != nullptr, "Static shape inference lacks constant data on port 0");
    }
    output_shapes.emplace_back(ov::PartialShape{ov::Dimension::dynamic()});
}
}  // namespace util
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const StringTensorUnpack* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1);
    const auto& data_shape = input_shapes[0];
    auto output_shapes = std::vector<TRShape>{data_shape, data_shape};
    if (data_shape.is_static()) {
        const auto string_data = util::get_string_tensor(op, 0, tensor_accessor);
        if (string_data) {
            uint32_t string_count = string_data.get_size();
            const auto tensor_data = string_data.data<std::string>();
            uint32_t total_length = 0;
            for (size_t i = 0; i < string_count; ++i)
                total_length += (*(tensor_data + i)).length();
            output_shapes.emplace_back(TRShape{total_length});
        } else {
            util::handle_else_case<TRShape>(output_shapes, op);
        }
    } else {
        util::handle_else_case<TRShape>(output_shapes, op);
    }

    return output_shapes;
}
}  // namespace v15
}  // namespace op
}  // namespace ov
