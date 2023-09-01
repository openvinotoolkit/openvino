// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"

namespace ov {
namespace builder {

std::shared_ptr<ov::Node> makeSpaceToDepth(const ov::Output<Node> &in,
                                               ov::opset3::SpaceToDepth::SpaceToDepthMode mode,
                                               size_t blockSize) {
    auto dtsNode = std::make_shared<ov::opset3::SpaceToDepth>(in, mode, blockSize);
    return dtsNode;
}

}  // namespace builder
}  // namespace ov
