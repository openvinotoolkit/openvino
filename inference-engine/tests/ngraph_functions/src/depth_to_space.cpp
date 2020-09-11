// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeDepthToSpace(const ngraph::Output<Node> &in,
                                               ngraph::opset3::DepthToSpace::DepthToSpaceMode mode,
                                               size_t blockSize) {
    auto dtsNode = std::make_shared<ngraph::opset3::DepthToSpace>(in, mode, blockSize);
    return dtsNode;
}

}  // namespace builder
}  // namespace ngraph