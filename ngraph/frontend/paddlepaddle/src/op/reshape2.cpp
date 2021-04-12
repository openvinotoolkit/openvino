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
#include "reshape2.hpp"
#include "utility.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {
    
OutputVector reshape2(const NodeContext& node) {
    auto data = node.get_ng_input("X");
    if (!node.has_ng_input("Shape") && !node.has_ng_input("ShapeTensor"))
    {
        auto shape_attr = node.get_attribute<std::vector<int32_t>>("shape");
        auto shape_node = ngraph::opset6::Constant::create(ngraph::element::i32, {shape_attr.size()}, shape_attr);
        return {std::make_shared<ngraph::opset6::Reshape>(data, shape_node, true)};
    } else {
        NOT_IMPLEMENTED("reshape2 with shape as input");
    }
}

} // namespace op
} // namespace pdpd
} // namespace frontend
} // namespace ngraph