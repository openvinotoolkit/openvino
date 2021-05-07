// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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