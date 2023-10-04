// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/concat.hpp"

#include <memory>
#include <vector>

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeConcat(const std::vector<ov::Output<Node>>& in, const int& axis) {
    return std::make_shared<ov::op::v0::Concat>(in, axis);
}

}  // namespace builder
}  // namespace ngraph
