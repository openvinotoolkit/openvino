// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/shuffle_channels.hpp"

#include <memory>
#include <vector>

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeShuffleChannels(const ov::Output<Node>& in, int axis, int group) {
    return std::make_shared<ov::op::v0::ShuffleChannels>(in, axis, group);
}

}  // namespace builder
}  // namespace ngraph
