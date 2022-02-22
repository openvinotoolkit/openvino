// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeSpaceToDepth(const ngraph::Output<Node> &in,
                                               ngraph::opset3::SpaceToDepth::SpaceToDepthMode mode,
                                               size_t blockSize) {
    auto dtsNode = std::make_shared<ngraph::opset3::SpaceToDepth>(in, mode, blockSize);
    return dtsNode;
}

}  // namespace builder
}  // namespace ngraph