// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <node_context.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs exp(const NodeContext& node) {
    auto data = node.get_ng_input("X");
    PDPD_OP_VALIDATION_CHECK(node, data.get_element_type() != ov::element::f64, "exp not support fp64 input type !");
    return node.default_single_output_mapping({std::make_shared<opset8::Exp>(data)}, {"Out"});
}
}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ngraph