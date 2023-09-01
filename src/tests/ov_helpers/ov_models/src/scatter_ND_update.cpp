// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"

namespace ov {
namespace builder {

std::shared_ptr<ov::Node> makeScatterNDUpdate(const ov::Output<Node> &in,
                                                  const element::Type& indicesType,
                                                  const std::vector<size_t>& indicesShape,
                                                  const std::vector<size_t>& indices,
                                                  const ov::Output<Node> &update) {
    auto indicesNode = std::make_shared<ov::opset1::Constant>(indicesType, indicesShape, indices);
    auto dtsNode = std::make_shared<ov::opset4::ScatterNDUpdate>(in, indicesNode, update);
    return dtsNode;
}

}  // namespace builder
}  // namespace ov
