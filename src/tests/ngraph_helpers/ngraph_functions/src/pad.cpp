// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
std::shared_ptr<ngraph::Node> makePad(const ngraph::Output<Node>& data,
                                      const std::vector<int64_t>& padsBegin,
                                      const std::vector<int64_t>& padsEnd,
                                      float argPadValue,
                                      ngraph::helpers::PadMode padMode) {
    ngraph::op::PadMode pad_mode;
    switch (padMode) {
    case ngraph::helpers::PadMode::CONSTANT:
        pad_mode = ngraph::op::PadMode::CONSTANT;
        break;
    case ngraph::helpers::PadMode::EDGE:
        pad_mode = ngraph::op::PadMode::EDGE;
        break;
    case ngraph::helpers::PadMode::REFLECT:
        pad_mode = ngraph::op::PadMode::REFLECT;
        break;
    case ngraph::helpers::PadMode::SYMMETRIC:
        pad_mode = ngraph::op::PadMode::SYMMETRIC;
        break;
    default:
        throw std::runtime_error("Can't create layer for this pad mode");
    }

    auto pads_begin = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64,
                                                                 ngraph::Shape{padsBegin.size()}, padsBegin.data());
    auto pads_end = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64,
                                                               ngraph::Shape{padsEnd.size()}, padsEnd.data());
    auto arg_pad_value = std::make_shared<ngraph::opset3::Constant>(data.get_element_type(), ngraph::Shape{}, &argPadValue);
    return std::make_shared<ngraph::opset3::Pad>(data, pads_begin, pads_end, arg_pad_value, pad_mode);
}

std::shared_ptr<ov::Node> makePad(const ov::Output<Node>& in,
                                  const ov::Output<Node>& beginNode,
                                  const ov::Output<Node>& endNode,
                                  const ov::Output<Node>& valueNode,
                                  ngraph::helpers::PadMode padMode) {
    ngraph::op::PadMode pad_mode;
    switch (padMode) {
    case ngraph::helpers::PadMode::CONSTANT:
        pad_mode = ngraph::op::PadMode::CONSTANT;
        break;
    case ngraph::helpers::PadMode::EDGE:
        pad_mode = ngraph::op::PadMode::EDGE;
        break;
    case ngraph::helpers::PadMode::REFLECT:
        pad_mode = ngraph::op::PadMode::REFLECT;
        break;
    case ngraph::helpers::PadMode::SYMMETRIC:
        pad_mode = ngraph::op::PadMode::SYMMETRIC;
        break;
    default:
        throw std::runtime_error("Can't create layer for this pad mode");
    }
    if (valueNode.get_node_shared_ptr() == nullptr)
        return std::make_shared<ov::op::v1::Pad>(in, beginNode, endNode, pad_mode);
    else
        return std::make_shared<ov::op::v1::Pad>(in, beginNode, endNode, valueNode, pad_mode);
}

}  // namespace builder
}  // namespace ngraph
