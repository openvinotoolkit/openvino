// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include "assign_value.hpp"
namespace ngraph {
    namespace frontend {
        namespace pdpd {
            namespace op {

                NamedOutputs assign_value (const NodeContext& node) {

                    std::vector<int32_t> shape = node.get_attribute<std::vector<int32_t>>("shape");
                    auto dtype = node.get_attribute<ngraph::element::Type>("dtype");
                    std::shared_ptr<Node> const_node;
                    PDPD_ASSERT(dtype != element::f64, "PDPD 2.0 doesn't support FLOAT64 yet");
                    switch (dtype) {
                        case element::i32:
                        {
                            auto values = node.get_attribute<std::vector<int32_t>>("int32_values");
                            const_node = {opset6::Constant::create(dtype, Shape{shape.begin(), shape.end()}, values)};
                            break;
                        }
                        case element::f32:
                        {
                            std::vector<float> values = node.get_attribute<std::vector<float>>("fp32_values");
                            const_node = {opset6::Constant::create(dtype, Shape{shape.begin(), shape.end()}, values)};
                            break;
                        }
                        case element::boolean:
                        {
                            auto values = node.get_attribute<std::vector<int32_t>>("bool_values");
                            const_node = {opset6::Constant::create(dtype, Shape{shape.begin(), shape.end()}, values)};
                            break;
                        }
                        default:
                        {
                            auto values = node.get_attribute<std::vector<int64_t>>("int64_values");
                            const_node = {opset6::Constant::create(dtype, Shape{shape.begin(), shape.end()}, values)};
                            break;
                        }
                    }

                    return node.default_single_output_mapping({const_node}, {"Out"});
                }

            }
        }
    }
}
