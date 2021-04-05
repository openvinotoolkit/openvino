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
#include "softmax.hpp"
#include "utility.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {
    OutputVector softmax(const NodeContext& node) {
        auto data = node.get_ng_input("X");
        auto axis = node.get_attribute<int32_t>("axis");
        if (axis < 0)
        {
            MY_ASSERT(data.get_partial_shape().rank().is_static(), "Softmax rank must be static");
            auto data_rank = data.get_partial_shape().rank().get_length();
            axis = data_rank + axis;
        }
        return {std::make_shared<ngraph::opset6::Softmax>(data, axis)};
    }
} // namespace op
} // namespace pdpd
} // namespace frontend
} // namespace ngraph