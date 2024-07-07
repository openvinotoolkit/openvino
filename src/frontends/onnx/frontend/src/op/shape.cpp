// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {

namespace opset_15 {

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

ONNX_OP("Shape", OPSET_SINCE(15), ai_onnx::opset_15::shape);
}  // namespace opset_15

namespace opset_1 {

ov::OutputVector shape(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);
    return {std::make_shared<v3::ShapeOf>(data)};
}

ONNX_OP("Shape", OPSET_RANGE(1, 14), ai_onnx::opset_1::shape);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
