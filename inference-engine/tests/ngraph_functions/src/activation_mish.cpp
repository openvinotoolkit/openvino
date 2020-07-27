// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <vector>
#include <memory>

#include "ngraph_functions/utils/ngraph_helpers.hpp"


namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeActivationMish(const ngraph::Output<Node> &in) {
    return std::make_shared<ngraph::op::v4::Mish>(in);
}

}  // namespace builder
}  // namespace ngraph
