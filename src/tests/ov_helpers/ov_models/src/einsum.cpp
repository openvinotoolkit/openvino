// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/einsum.hpp"

#include <memory>
#include <string>

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeEinsum(const OutputVector& inputs, const std::string& equation) {
    std::shared_ptr<ov::Node> einsum = std::make_shared<ov::op::v7::Einsum>(inputs, equation);
    return einsum;
}

}  // namespace builder
}  // namespace ngraph
