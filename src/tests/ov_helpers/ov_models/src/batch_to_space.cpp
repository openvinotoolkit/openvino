// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/batch_to_space.hpp"

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeBatchToSpace(const ov::Output<Node>& in,
                                           const element::Type& type,
                                           const std::vector<int64_t>& blockShape,
                                           const std::vector<int64_t>& cropsBegin,
                                           const std::vector<int64_t>& cropsEnd) {
    ov::Shape constShape = {in.get_partial_shape().size()};
    auto blockShapeNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, blockShape.data());
    auto cropsBeginNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, cropsBegin.data());
    auto cropsEndNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, cropsEnd.data());

    auto btsNode = std::make_shared<ov::op::v1::BatchToSpace>(in, blockShapeNode, cropsBeginNode, cropsEndNode);
    return btsNode;
}

}  // namespace builder
}  // namespace ngraph
