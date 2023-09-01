// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ov_models/builders.hpp"

namespace ov {
namespace builder {

std::shared_ptr<Node> makeShuffleChannels(const ov::Output<Node> &in,
                                          int axis,
                                          int group) {
    return std::make_shared<ov::opset3::ShuffleChannels>(in, axis, group);
}

}  // namespace builder
}  // namespace ov
