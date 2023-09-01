// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>

#include "ov_models/builders.hpp"

namespace ov {
namespace builder {

    std::shared_ptr<ov::Node> makeEinsum(const OutputVector& inputs,
                                             const std::string& equation) {
        std::shared_ptr<ov::Node> einsum = std::make_shared<ov::opset7::Einsum>(inputs, equation);
        return einsum;
    }

}  // namespace builder
}  // namespace ov
