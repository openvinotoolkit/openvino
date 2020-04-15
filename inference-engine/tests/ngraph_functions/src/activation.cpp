// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <vector>
#include <memory>

#include "ngraph_functions/utils/ngraph_helpers.hpp"


namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeActivation(const ngraph::Output<Node> &in,
                                             const element::Type &type,
                                             ngraph::helpers::ActivationTypes activationType) {
    auto leaky_slope = std::make_shared<ngraph::op::Constant>(
            ngraph::element::f32,
            ngraph::Shape{1},
            std::vector<float>{0.01f});

    switch (activationType) {
        case ngraph::helpers::ActivationTypes::Sigmoid:
            return std::make_shared<ngraph::op::Sigmoid>(in);
        case ngraph::helpers::ActivationTypes::Tanh:
            return std::make_shared<ngraph::op::Tanh>(in);
        case ngraph::helpers::ActivationTypes::Relu:
            return std::make_shared<ngraph::op::Relu>(in);
        case ngraph::helpers::ActivationTypes::LeakyRelu:
            return std::make_shared<ngraph::op::PRelu>(in, leaky_slope);
        case ngraph::helpers::ActivationTypes::Exp:
            return std::make_shared<ngraph::op::Exp>(in);
        case ngraph::helpers::ActivationTypes::Log:
            return std::make_shared<ngraph::op::Log>(in);
        case ngraph::helpers::ActivationTypes::Sign:
            return std::make_shared<ngraph::op::Sign>(in);
        case ngraph::helpers::ActivationTypes::Abs:
            return std::make_shared<ngraph::op::Abs>(in);
        default:
            throw std::runtime_error("Can't create layer for this activation type");
    }
}

}  // namespace builder
}  // namespace ngraph
