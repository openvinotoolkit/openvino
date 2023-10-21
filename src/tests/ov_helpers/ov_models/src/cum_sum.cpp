// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/cum_sum.hpp"

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeCumSum(const ov::Output<Node>& in,
                                     const ov::Output<Node>& axis,
                                     bool exclusive,
                                     bool reverse) {
    return std::make_shared<ov::op::v0::CumSum>(in, axis, exclusive, reverse);
}

}  // namespace builder
}  // namespace ngraph
