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
                                             ngraph::helpers::ActivationTypes activationType,
                                             std::vector<size_t> inShape) {
    switch (activationType) {
        case ngraph::helpers::ActivationTypes::Sigmoid:
            return std::make_shared<ngraph::op::v0::Sigmoid>(in);
        case ngraph::helpers::ActivationTypes::Tanh:
            return std::make_shared<ngraph::op::v0::Tanh>(in);
        case ngraph::helpers::ActivationTypes::Relu:
            return std::make_shared<ngraph::op::v0::Relu>(in);
        case ngraph::helpers::ActivationTypes::LeakyRelu: {
            auto leaky_slope = std::make_shared<ngraph::op::v0::Constant>(
                    ngraph::element::f32,
                    inShape,
                    std::vector<float>{0.01f});
            return std::make_shared<ngraph::op::v0::PRelu>(in, leaky_slope);
        }
        case ngraph::helpers::ActivationTypes::Exp:
            return std::make_shared<ngraph::op::v0::Exp>(in);
        case ngraph::helpers::ActivationTypes::Log:
            return std::make_shared<ngraph::op::v0::Log>(in);
        case ngraph::helpers::ActivationTypes::Sign:
            return std::make_shared<ngraph::op::v0::Sign>(in);
        case ngraph::helpers::ActivationTypes::Abs:
            return std::make_shared<ngraph::op::v0::Abs>(in);
        case ngraph::helpers::ActivationTypes::Gelu:
            return std::make_shared<ngraph::op::v0::Gelu>(in);
        case ngraph::helpers::ActivationTypes::Clamp:
            return std::make_shared<ngraph::op::v0::Clamp>(in, -2.0, 2.0);
        case ngraph::helpers::ActivationTypes::Negative:
            return std::make_shared<ngraph::op::v0::Negative>(in);
        case ngraph::helpers::ActivationTypes::Acos:
            return std::make_shared<ngraph::op::v0::Acos>(in);
        case ngraph::helpers::ActivationTypes::Asin:
            return std::make_shared<ngraph::op::v0::Asin>(in);
        case ngraph::helpers::ActivationTypes::Atan:
            return std::make_shared<ngraph::op::v0::Atan>(in);
        case ngraph::helpers::ActivationTypes::Cos:
            return std::make_shared<ngraph::op::v0::Cos>(in);
        case ngraph::helpers::ActivationTypes::Cosh:
            return std::make_shared<ngraph::op::v0::Cosh>(in);
        case ngraph::helpers::ActivationTypes::Floor:
            return std::make_shared<ngraph::op::v0::Floor>(in);
        case ngraph::helpers::ActivationTypes::Sin:
            return std::make_shared<ngraph::op::v0::Sin>(in);
        case ngraph::helpers::ActivationTypes::Sinh:
            return std::make_shared<ngraph::op::v0::Sinh>(in);
        case ngraph::helpers::ActivationTypes::Sqrt:
            return std::make_shared<ngraph::op::v0::Sqrt>(in);
        case ngraph::helpers::ActivationTypes::Tan:
            return std::make_shared<ngraph::op::v0::Tan>(in);
        case ngraph::helpers::ActivationTypes::Elu:
            return std::make_shared<ngraph::op::v0::Elu>(in, 0.1);
        case ngraph::helpers::ActivationTypes::Erf:
            return std::make_shared<ngraph::op::v0::Erf>(in);
        case ngraph::helpers::ActivationTypes::HardSigmoid: {
            auto hard_sigmoid_alpha = std::make_shared<ngraph::op::v0::Constant>(
                    type, inShape, 0.2f);
            auto hard_sigmoid_beta = std::make_shared<ngraph::op::v0::Constant>(
                    type, inShape, 0.5f);
            return std::make_shared<ngraph::op::v0::HardSigmoid>(in, hard_sigmoid_alpha, hard_sigmoid_beta);
        }
        case ngraph::helpers::ActivationTypes::Selu: {
            auto selu_alpha = std::make_shared<ngraph::op::v0::Constant>(
                    type, inShape, 1.6732f);
            auto selu_lambda = std::make_shared<ngraph::op::v0::Constant>(
                    type, inShape, 1.0507f);
            return std::make_shared<ngraph::op::v0::Selu>(in, selu_alpha, selu_lambda);
        }
        case ngraph::helpers::ActivationTypes::Ceiling:
            return std::make_shared<ngraph::op::v0::Ceiling>(in);
        case ngraph::helpers::ActivationTypes::PReLu: {
            auto negative_slope = std::make_shared<ngraph::op::v0::Constant>(
                    ngraph::element::f32,
                    inShape,
                    std::vector<float>{-0.01f});
            return std::make_shared<ngraph::op::v0::PRelu>(in, negative_slope);
        }
        case ngraph::helpers::ActivationTypes::Mish:
            return std::make_shared<ngraph::op::v4::Mish>(in);
        default:
            throw std::runtime_error("Can't create layer for this activation type");
    }
}

std::shared_ptr<ngraph::Node> makeActivation(const ngraph::ParameterVector &parameters,
                                             const element::Type &type,
                                             ngraph::helpers::ActivationTypes activationType) {
    switch (activationType) {
        case ngraph::helpers::ActivationTypes::LeakyRelu:
            return std::make_shared<ngraph::op::v0::PRelu>(parameters[0], parameters[1]);
        case ngraph::helpers::ActivationTypes::HardSigmoid:
            return std::make_shared<ngraph::op::v0::HardSigmoid>(parameters[0], parameters[1], parameters[2]);
        case ngraph::helpers::ActivationTypes::Selu:
            return std::make_shared<ngraph::op::v0::Selu>(parameters[0], parameters[1], parameters[2]);
        case ngraph::helpers::ActivationTypes::PReLu:
            return std::make_shared<ngraph::op::v0::PRelu>(parameters[0], parameters[1]);
        default:
            throw std::runtime_error("It is impossible to create layer for this activation type with input as parameter");
    }
}

}  // namespace builder
}  // namespace ngraph
