// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeCumSum(const ngraph::Output<Node> &in,
                                         const ngraph::Output<Node> &axis,
                                         bool exclusive,
                                         bool reverse) {
    return std::make_shared<ngraph::op::CumSum>(in, axis, exclusive, reverse);
}

}  // namespace builder
}  // namespace ngraph