// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/variadic_split.hpp"

#include <memory>
#include <vector>

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {
std::shared_ptr<ov::Node> makeVariadicSplit(const ov::Output<Node>& in,
                                            const std::vector<size_t> numSplits,
                                            int64_t axis) {
    auto splitAxisOp = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{}, std::vector<int64_t>{axis});
    auto numSplit = std::make_shared<ov::op::v0::Constant>(element::u64, ov::Shape{numSplits.size()}, numSplits);
    auto VariadicSplitNode = std::make_shared<ov::op::v1::VariadicSplit>(in, splitAxisOp, numSplit);
    return VariadicSplitNode;
}
}  // namespace builder
}  // namespace ngraph
