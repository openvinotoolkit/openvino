//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <ngraph/opsets/opset6.hpp>
#include "scale.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

    NamedOutputs scale (const NodeContext& node) {
        auto data = node.get_ng_input("X");
        auto scale = ngraph::opset6::Constant::create(ngraph::element::f32, {1}, {node.get_attribute<float>("scale")});
        auto bias = ngraph::opset6::Constant::create(ngraph::element::f32, {1}, {node.get_attribute<float>("bias")});
        auto bias_after_scale = node.get_attribute<bool>("bias_after_scale");
        auto fp32_data = std::make_shared<ngraph::opset6::Convert>(data, element::f32);
        if(!bias_after_scale) {
            auto node_add = std::make_shared<ngraph::opset6::Add>(fp32_data, bias);
            return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Multiply>(node_add, scale)}, {"Out"});
        } else {
            auto node_multiply =  std::make_shared<ngraph::opset6::Multiply>(fp32_data, scale);
            return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Add>(node_multiply, bias)}, {"Out"});
        }
    }

}}}}