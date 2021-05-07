// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include "fill_constant.hpp"


namespace ngraph {
    namespace frontend {
        namespace pdpd {
            namespace op {

                NamedOutputs fill_constant (const NodeContext& node) {

                    auto shape = node.get_attribute<std::vector<int64_t>>("shape");
                    auto dtype = node.get_attribute<ngraph::element::Type>("dtype");
                    //TODO to Support Tensor/Tuple Input add more tests for other data types
                    Output<Node> value_node;
                    if(dtype == element::i32) {
                        int32_t  value = node.get_attribute<int32_t>("value");
                        value_node = opset6::Constant::create(dtype, {1}, {value});
                    }
                    else if(dtype == element::f32) {
                        float value = node.get_attribute<float>("value");
                        value_node = opset6::Constant::create(dtype, {1}, {value});
                    }

                    auto shape_node = opset6::Constant::create(element::i64, {shape.size()}, shape);
                    return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Broadcast>(value_node, shape_node)}, {"Out"});
                }

            }}}}