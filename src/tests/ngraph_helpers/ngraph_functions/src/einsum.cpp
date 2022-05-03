// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

    std::shared_ptr<ngraph::Node> makeEinsum(const OutputVector& inputs,
                                             const std::string& equation) {
        std::shared_ptr<ngraph::Node> einsum = std::make_shared<ngraph::opset7::Einsum>(inputs, equation);
        return einsum;
    }

}  // namespace builder
}  // namespace ngraph
