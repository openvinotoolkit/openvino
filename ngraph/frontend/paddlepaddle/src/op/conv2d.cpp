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
#include "conv2d.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

OutputVector conv2d (const NodeContext& node) {
    auto data = node.get_ng_input("Input");
    auto filter = node.get_ng_input("Filter");
    // TODO: resolve padding according to spec
    auto strides = node.get_attribute<std::vector<int32_t>>("strides");
    auto paddings = node.get_attribute<std::vector<int32_t>>("paddings");
    auto dilations = node.get_attribute<std::vector<int32_t>>("dilations");
    return {std::make_shared<ngraph::opset6::Convolution>(
        data,
        filter,
        ngraph::Strides(strides.begin(), strides.end()),
        ngraph::CoordinateDiff(paddings.begin(), paddings.end()),
        ngraph::CoordinateDiff(paddings.begin(), paddings.end()),
        ngraph::Strides(dilations.begin(), dilations.end()))};
}

}}}}