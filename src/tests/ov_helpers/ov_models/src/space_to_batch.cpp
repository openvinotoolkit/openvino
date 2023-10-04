// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/space_to_batch.hpp"

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeSpaceToBatch(const ov::Output<Node>& in,
                                           const element::Type& type,
                                           const std::vector<int64_t>& blockShape,
                                           const std::vector<int64_t>& padsBegin,
                                           const std::vector<int64_t>& padsEnd) {
    ov::Shape constShape = {in.get_partial_shape().size()};

    auto blockShapeNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, blockShape.data());
    auto padsBeginNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, padsBegin.data());
    auto padsEndNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, padsEnd.data());
    auto stbNode = std::make_shared<ov::op::v1::SpaceToBatch>(in, blockShapeNode, padsBeginNode, padsEndNode);
    return stbNode;
}

}  // namespace builder
}  // namespace ngraph
