// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/squeeze.hpp"

#include <memory>
#include <vector>

#include "openvino/op/unsqueeze.hpp"
#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {
std::shared_ptr<ov::Node> makeSqueezeUnsqueeze(const ov::Output<Node>& in,
                                               const element::Type& type,
                                               const std::vector<int>& squeeze_indices,
                                               ov::test::utils::SqueezeOpType opType) {
    auto constant = std::make_shared<ov::op::v0::Constant>(type, ov::Shape{squeeze_indices.size()}, squeeze_indices);
    switch (opType) {
    case ov::test::utils::SqueezeOpType::SQUEEZE:
        return std::make_shared<ov::op::v0::Squeeze>(in, constant);
    case ov::test::utils::SqueezeOpType::UNSQUEEZE:
        return std::make_shared<ov::op::v0::Unsqueeze>(in, constant);
    default:
        throw std::logic_error("Unsupported operation type");
    }
}
}  // namespace builder
}  // namespace ngraph
