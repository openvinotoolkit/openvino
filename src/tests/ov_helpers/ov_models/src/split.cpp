// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ov_models/builders.hpp"

namespace ov {
namespace builder {
std::shared_ptr<ov::Node> makeSplit(const ov::Output<Node> &in,
                                        const element::Type &type,
                                        size_t numSplits,
                                        int64_t axis) {
    auto splitAxisOp = std::make_shared<ov::opset1::Constant>(ov::element::Type_t::i64, ov::Shape{},
                                                                  std::vector<int64_t>{axis});
    auto splitNode = std::make_shared<ov::opset1::Split>(in, splitAxisOp, numSplits);
    return splitNode;
}
}  // namespace builder
}  // namespace ov
