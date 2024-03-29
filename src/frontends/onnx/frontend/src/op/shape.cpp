// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/shape.hpp"

#include <cstdint>

#include "core/null_node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {

namespace set_15 {

ov::OutputVector shape(const ov::frontend::onnx::Node& node) {
    using ov::op::util::is_null;

    const auto data = node.get_ov_inputs().at(0);
    const auto input_shape = std::make_shared<v3::ShapeOf>(data);

    const auto start_val = node.get_attribute_value<int64_t>("start", 0);
    const auto end_val = node.get_attribute_value<int64_t>("end", std::numeric_limits<int64_t>::max());

    if (start_val == 0 && end_val == INT64_MAX) {
        return {input_shape};
    }

    const auto start = v0::Constant::create(element::i64, ov::Shape{1}, {start_val});
    const auto end = v0::Constant::create(element::i64, ov::Shape{1}, {end_val});
    const auto default_step = v0::Constant::create(element::i64, {1}, {1});

    return {std::make_shared<v8::Slice>(input_shape, start, end, default_step)};
}

}  // namespace set_15

namespace set_1 {

ov::OutputVector shape(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);
    return {std::make_shared<v3::ShapeOf>(data)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
