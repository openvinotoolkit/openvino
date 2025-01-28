// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/range.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/constant.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {
ov::OutputVector range(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 2);
    auto nodes = node.get_ov_inputs();

    auto start = nodes[0];
    auto limit = nodes[1];
    auto delta =
        nodes.size() == 3 ? nodes[2] : ov::op::v0::Constant::create(start.get_element_type(), ov::Shape{}, {1});
    CHECK_VALID_NODE(node,
                     start.get_element_type() == limit.get_element_type(),
                     "start and limit must be of same type, got :",
                     start.get_element_type(),
                     limit.get_element_type());
    CHECK_VALID_NODE(node,
                     start.get_element_type() == delta.get_element_type(),
                     "start and delta must be of same type, got :",
                     start.get_element_type(),
                     delta.get_element_type());
    return {std::make_shared<ov::op::v4::Range>(start, limit, delta, start.get_element_type())};
}
ONNX_OP("Range", OPSET_SINCE(1), com_microsoft::opset_1::range, MICROSOFT_DOMAIN);
}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov