// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"


namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeInputLayer(const element::Type &type, ngraph::helpers::InputLayerType inputType,
                                             const std::vector<size_t> &shape) {
    std::shared_ptr<ngraph::Node> input;
    switch (inputType) {
        case ngraph::helpers::InputLayerType::CONSTANT: {
            input = ngraph::builder::makeConstant<float>(type, shape, {}, true);
            break;
        }
        case ngraph::helpers::InputLayerType::PARAMETER:
            input = ngraph::builder::makeParams(type, {shape})[0];
            break;
        default:
           throw std::runtime_error("Unsupported inputType");
    }
    return input;
}
}  // namespace builder
}  // namespace ngraph