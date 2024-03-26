// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/shape.hpp"

#include <cstdint>

#include "core/null_node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/broadcast.hpp"
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
    ov::OutputVector inputs{node.get_ov_inputs()};
    const auto& data = inputs.at(0);
    auto input_shape = std::make_shared<v3::ShapeOf>(data);

    auto start_val = node.get_attribute_value<int64_t>("start", 0);
    auto end_val = node.get_attribute_value<int64_t>("end", INT64_MAX);
    start_val = start_val < 0 ? -1 : start_val;
    end_val = end_val < 0 ? -1 : end_val;
    auto start = v0::Constant::create(element::i64, ov::Shape{1}, {start_val});
    auto end = v0::Constant::create(element::i64, ov::Shape{1}, {end_val});
    const auto& default_step = v0::Constant::create(element::i64, {1}, {1});
    ov::Output<ov::Node> steps =
        std::make_shared<v3::Broadcast>(default_step, std::make_shared<v3::ShapeOf>(start, element::i64));
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
