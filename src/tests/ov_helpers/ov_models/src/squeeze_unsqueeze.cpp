// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ov_models/builders.hpp"

namespace ov {
namespace builder {
std::shared_ptr<ov::Node> makeSqueezeUnsqueeze(const ov::Output<Node> &in,
                                                   const element::Type &type,
                                                   const std::vector<int> &squeeze_indices,
                                                   ov::helpers::SqueezeOpType opType) {
    auto constant = std::make_shared<ov::opset1::Constant>(type, ov::Shape{squeeze_indices.size()}, squeeze_indices);
    switch (opType) {
        case ov::helpers::SqueezeOpType::SQUEEZE:
            return std::make_shared<ov::opset1::Squeeze>(in, constant);
        case ov::helpers::SqueezeOpType::UNSQUEEZE:
            return std::make_shared<ov::opset1::Unsqueeze>(in, constant);
        default:
            throw std::logic_error("Unsupported operation type");
    }
}
} // namespace builder
} // namespace ov
