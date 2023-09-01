// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ov_models/builders.hpp"

namespace ov {
namespace builder {
std::shared_ptr<ov::Node> makePad(const ov::Output<Node>& data,
                                      const std::vector<int64_t>& padsBegin,
                                      const std::vector<int64_t>& padsEnd,
                                      float argPadValue,
                                      ov::helpers::PadMode padMode) {
    ov::op::PadMode pad_mode;
    switch (padMode) {
    case ov::helpers::PadMode::CONSTANT:
        pad_mode = ov::op::PadMode::CONSTANT;
        break;
    case ov::helpers::PadMode::EDGE:
        pad_mode = ov::op::PadMode::EDGE;
        break;
    case ov::helpers::PadMode::REFLECT:
        pad_mode = ov::op::PadMode::REFLECT;
        break;
    case ov::helpers::PadMode::SYMMETRIC:
        pad_mode = ov::op::PadMode::SYMMETRIC;
        break;
    default:
        throw std::runtime_error("Can't create layer for this pad mode");
    }

    auto pads_begin = std::make_shared<ov::opset3::Constant>(ov::element::i64,
                                                                 ov::Shape{padsBegin.size()}, padsBegin.data());
    auto pads_end = std::make_shared<ov::opset3::Constant>(ov::element::i64,
                                                               ov::Shape{padsEnd.size()}, padsEnd.data());
    auto arg_pad_value = std::make_shared<ov::opset3::Constant>(data.get_element_type(), ov::Shape{}, &argPadValue);
    return std::make_shared<ov::opset3::Pad>(data, pads_begin, pads_end, arg_pad_value, pad_mode);
}

std::shared_ptr<ov::Node> makePad(const ov::Output<Node>& in,
                                  const ov::Output<Node>& beginNode,
                                  const ov::Output<Node>& endNode,
                                  const ov::Output<Node>& valueNode,
                                  ov::helpers::PadMode padMode) {
    ov::op::PadMode pad_mode;
    switch (padMode) {
    case ov::helpers::PadMode::CONSTANT:
        pad_mode = ov::op::PadMode::CONSTANT;
        break;
    case ov::helpers::PadMode::EDGE:
        pad_mode = ov::op::PadMode::EDGE;
        break;
    case ov::helpers::PadMode::REFLECT:
        pad_mode = ov::op::PadMode::REFLECT;
        break;
    case ov::helpers::PadMode::SYMMETRIC:
        pad_mode = ov::op::PadMode::SYMMETRIC;
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
}  // namespace ov
