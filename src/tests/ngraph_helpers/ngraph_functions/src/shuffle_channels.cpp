// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeShuffleChannels(const ngraph::Output<Node> &in,
                                          int axis,
                                          int group) {
    return std::make_shared<ngraph::opset3::ShuffleChannels>(in, axis, group);
}

}  // namespace builder
}  // namespace ngraph