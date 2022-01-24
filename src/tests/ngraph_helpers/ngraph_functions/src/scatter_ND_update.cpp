// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeScatterNDUpdate(const ngraph::Output<Node> &in,
                                                  const element::Type& indicesType,
                                                  const std::vector<size_t>& indicesShape,
                                                  const std::vector<size_t>& indices,
                                                  const ngraph::Output<Node> &update) {
    auto indicesNode = std::make_shared<ngraph::opset1::Constant>(indicesType, indicesShape, indices);
    auto dtsNode = std::make_shared<ngraph::opset4::ScatterNDUpdate>(in, indicesNode, update);
    return dtsNode;
}

}  // namespace builder
}  // namespace ngraph