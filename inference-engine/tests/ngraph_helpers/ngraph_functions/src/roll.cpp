// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeRoll(const ngraph::Output<Node> &in,
                                       const ngraph::Output<Node> &shift,
                                       const ngraph::Output<Node> &axes) {
    return std::make_shared<ngraph::op::v7::Roll>(in, shift, axes);
}

}  // namespace builder
}  // namespace ngraph
