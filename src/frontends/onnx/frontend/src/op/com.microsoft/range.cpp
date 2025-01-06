// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/frontend/exception.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {
ov::OutputVector range(const ov::frontend::onnx::Node& node) {
    auto nodes = node.get_ov_inputs();
    FRONT_END_GENERAL_CHECK((nodes.size() == 2 || nodes.size() == 3), "Range takes 2 or 3 inputs. Provided " + std::to_string(nodes.size()));
    auto start = nodes.at(0);
    auto limit = nodes.at(1);
    auto step = nodes.size() == 3 ? nodes.at(2) : ov::op::v0::Constant::create(start.get_element_type(), ov::Shape{}, {1});
    return {std::make_shared<ov::op::v4::Range>(start, limit, step,start.get_element_type())};
}
ONNX_OP("Range", OPSET_SINCE(1), com_microsoft::opset_1::range, MICROSOFT_DOMAIN);
}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov