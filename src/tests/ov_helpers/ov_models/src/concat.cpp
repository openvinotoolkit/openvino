// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ov_models/builders.hpp"

namespace ov {
namespace builder {

std::shared_ptr<ov::Node> makeConcat(const std::vector<ov::Output<Node>>& in, const int& axis) {
    return std::make_shared<ov::opset4::Concat>(in, axis);
}

}  // namespace builder
}  // namespace ov
