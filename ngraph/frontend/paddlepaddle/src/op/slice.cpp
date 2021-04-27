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
#include "slice.hpp"
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs slice (const NodeContext& node) {
    auto data = node.get_ng_input("Input");
    auto axes = node.get_attribute<std::vector<int32_t>>("axes");
    auto starts = node.get_attribute<std::vector<int32_t>>("starts");
    auto ends = node.get_attribute<std::vector<int32_t>>("ends");
    auto parialShape = data.get_partial_shape();
    PDPD_ASSERT(parialShape.is_static(), "slice: must use static shape.");
    auto shape = parialShape.to_shape();
    std::vector<int32_t> fixedStarts(shape.size(), 0);
    std::vector<int32_t> fixedEnds(shape.size(), INT_MAX);
    int n = 0;
    for (size_t &&i : axes) {
        PDPD_ASSERT(i < shape.size(), "slice: axes must be less than the X rank.");
        fixedStarts[i] = starts[n];
        fixedEnds[i] = ends[n];
        n++;
    }
    
    auto startsNode = ngraph::opset6::Constant::create(ngraph::element::i32, { shape.size() }, fixedStarts);
    auto endsNode = ngraph::opset6::Constant::create(ngraph::element::i32, { shape.size() }, fixedEnds);
    auto stridesNode = ngraph::opset6::Constant::create(ngraph::element::i32, { shape.size() }, std::vector<int32_t>(shape.size(), 1));
    return node.default_single_output_mapping({std::make_shared<ngraph::opset6::StridedSlice>(data,
        startsNode, 
        endsNode, 
        stridesNode,
        std::vector<int64_t>(shape.size(), 0),
        std::vector<int64_t>(shape.size(), 0))}, {"Out"});
}

}}}}