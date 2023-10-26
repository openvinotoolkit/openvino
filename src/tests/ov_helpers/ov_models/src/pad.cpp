// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/pad.hpp"

#include <memory>
#include <vector>

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {
std::shared_ptr<ov::Node> makePad(const ov::Output<Node>& data,
                                  const std::vector<int64_t>& padsBegin,
                                  const std::vector<int64_t>& padsEnd,
                                  float argPadValue,
                                  ov::test::utils::PadMode padMode,
                                  const bool allow_negative_pad) {
    ov::op::PadMode pad_mode;
    switch (padMode) {
    case ov::test::utils::PadMode::CONSTANT:
        pad_mode = ov::op::PadMode::CONSTANT;
        break;
    case ov::test::utils::PadMode::EDGE:
        pad_mode = ov::op::PadMode::EDGE;
        break;
    case ov::test::utils::PadMode::REFLECT:
        pad_mode = ov::op::PadMode::REFLECT;
        break;
    case ov::test::utils::PadMode::SYMMETRIC:
        pad_mode = ov::op::PadMode::SYMMETRIC;
        break;
    default:
        throw std::runtime_error("Can't create layer for this pad mode");
    }

    auto pads_begin =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{padsBegin.size()}, padsBegin.data());
    auto pads_end = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{padsEnd.size()}, padsEnd.data());
    auto arg_pad_value = std::make_shared<ov::op::v0::Constant>(data.get_element_type(), ov::Shape{}, &argPadValue);

    if (allow_negative_pad) {
        return std::make_shared<ov::op::v12::Pad>(data, pads_begin, pads_end, arg_pad_value, pad_mode);
    } else {
        return std::make_shared<ov::op::v1::Pad>(data, pads_begin, pads_end, arg_pad_value, pad_mode);
    }
}

std::shared_ptr<ov::Node> makePad(const ov::Output<Node>& in,
                                  const ov::Output<Node>& beginNode,
                                  const ov::Output<Node>& endNode,
                                  const ov::Output<Node>& valueNode,
                                  ov::test::utils::PadMode padMode,
                                  const bool allow_negative_pad) {
    ov::op::PadMode pad_mode;
    switch (padMode) {
    case ov::test::utils::PadMode::CONSTANT:
        pad_mode = ov::op::PadMode::CONSTANT;
        break;
    case ov::test::utils::PadMode::EDGE:
        pad_mode = ov::op::PadMode::EDGE;
        break;
    case ov::test::utils::PadMode::REFLECT:
        pad_mode = ov::op::PadMode::REFLECT;
        break;
    case ov::test::utils::PadMode::SYMMETRIC:
        pad_mode = ov::op::PadMode::SYMMETRIC;
        break;
    default:
        throw std::runtime_error("Can't create layer for this pad mode");
    }
    if (valueNode.get_node_shared_ptr() == nullptr) {
        if (allow_negative_pad) {
            return std::make_shared<ov::op::v12::Pad>(in, beginNode, endNode, pad_mode);
        } else {
            return std::make_shared<ov::op::v1::Pad>(in, beginNode, endNode, pad_mode);
        }
    } else {
        if (allow_negative_pad) {
            return std::make_shared<ov::op::v12::Pad>(in, beginNode, endNode, valueNode, pad_mode);
        } else {
            return std::make_shared<ov::op::v1::Pad>(in, beginNode, endNode, valueNode, pad_mode);
        }
    }
}

}  // namespace builder
}  // namespace ngraph
