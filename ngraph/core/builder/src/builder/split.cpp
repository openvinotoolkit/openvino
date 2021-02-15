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

#include "ngraph/builder/split.hpp"
#include "ngraph/opsets/opset1.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;

OutputVector builder::opset1::split(const Output<Node>& value,
                                    const std::vector<size_t>& split_lengths,
                                    int64_t axis)
{
    const auto axis_node = ngraph::opset1::Constant::create(element::i64, Shape{}, {axis});
    const auto split_lengths_node =
        ngraph::opset1::Constant::create(element::u64, Shape{split_lengths.size()}, split_lengths);
    const auto variadic_split =
        std::make_shared<ngraph::opset1::VariadicSplit>(value, axis_node, split_lengths_node);

    return variadic_split->outputs();
}

OutputVector builder::opset1::split(const Output<Node>& value, size_t num_splits, int64_t axis)
{
    const auto axis_node = ngraph::opset1::Constant::create(element::i64, Shape{}, {axis});
    const auto split = std::make_shared<ngraph::opset1::Split>(value, axis_node, num_splits);

    return split->outputs();
}
