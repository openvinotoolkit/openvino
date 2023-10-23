// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/roll.hpp"

#include <memory>

#include "openvino/core/node.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeRoll(const ov::Output<Node>& in,
                                   const ov::Output<Node>& shift,
                                   const ov::Output<Node>& axes) {
    return std::make_shared<ov::op::v7::Roll>(in, shift, axes);
}

}  // namespace builder
}  // namespace ngraph
