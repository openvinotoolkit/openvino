// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include <node_context.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs fill_any_like(const NodeContext& node) {
    auto x = node.get_ng_input("X");
    auto dtype = node.get_attribute<ngraph::element::Type>("dtype", element::undefined);
    auto value = node.get_attribute<float>("value");
    if (dtype == element::undefined) {
        // when type does not define, use the input type
        dtype = x.get_element_type();
    }
    auto supported_type = {element::i32, element::i64, element::f16, element::f32, element::f64};
    bool valid_type = std::any_of(supported_type.begin(), supported_type.end(), [dtype](const element::Type& type) {
        return dtype == type;
    });
    PDPD_ASSERT(valid_type, "fill_any_like only supports i32, i64, f16, f32, f64");
    auto value_node = opset6::Constant::create(dtype, {1}, {value});
    auto shape_node = std::make_shared<opset6::ShapeOf>(x);

    return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Broadcast>(value_node, shape_node)},
                                              {"Out"});
}

}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ngraph
