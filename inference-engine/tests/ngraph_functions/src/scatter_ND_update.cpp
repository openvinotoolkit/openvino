// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeScatterNDUpdate(const ngraph::Output<Node> &in,
                                                  const ngraph::Output<Node> &indices,
                                                  const ngraph::Output<Node> &update) {
    auto dtsNode = std::make_shared<ngraph::opset3::ScatterNDUpdate>(in, indices, update);
    return dtsNode;
}

}  // namespace builder
}  // namespace ngraph