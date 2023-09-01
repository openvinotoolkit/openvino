// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ov_models/utils/ov_helpers.hpp"


namespace ov {
namespace builder {

std::shared_ptr<ov::Node> makeActivation(const ov::Output<Node> &in,
                                             const element::Type &type,
                                             ov::helpers::ActivationTypes activationType,
                                             std::vector<size_t> inShape,
                                             std::vector<float> constantsValue) {
    switch (activationType) {
        case ov::helpers::ActivationTypes::Sigmoid:
            return std::make_shared<ov::op::v0::Sigmoid>(in);
        case ov::helpers::ActivationTypes::Tanh:
            return std::make_shared<ov::op::v0::Tanh>(in);
        case ov::helpers::ActivationTypes::Relu:
            return std::make_shared<ov::op::v0::Relu>(in);
        case ov::helpers::ActivationTypes::LeakyRelu: {
            auto leaky_slope = std::make_shared<ov::op::v0::Constant>(
                    ov::element::f32,
                    inShape,
                    constantsValue);
            return std::make_shared<ov::op::v0::PRelu>(in, leaky_slope);
        }
        case ov::helpers::ActivationTypes::Exp:
            return std::make_shared<ov::op::v0::Exp>(in);
        case ov::helpers::ActivationTypes::Log:
            return std::make_shared<ov::op::v0::Log>(in);
        case ov::helpers::ActivationTypes::Sign:
            return std::make_shared<ov::op::v0::Sign>(in);
        case ov::helpers::ActivationTypes::Abs:
            return std::make_shared<ov::op::v0::Abs>(in);
        case ov::helpers::ActivationTypes::Gelu:
            return std::make_shared<ov::op::v0::Gelu>(in);
        case ov::helpers::ActivationTypes::Clamp:
            return std::make_shared<ov::op::v0::Clamp>(in, constantsValue[0], constantsValue[1]);
        case ov::helpers::ActivationTypes::Negative:
            return std::make_shared<ov::op::v0::Negative>(in);
        case ov::helpers::ActivationTypes::Acos:
            return std::make_shared<ov::op::v0::Acos>(in);
        case ov::helpers::ActivationTypes::Acosh:
            return std::make_shared<ov::op::v3::Acosh>(in);
        case ov::helpers::ActivationTypes::Asin:
            return std::make_shared<ov::op::v0::Asin>(in);
        case ov::helpers::ActivationTypes::Asinh:
            return std::make_shared<ov::op::v3::Asinh>(in);
        case ov::helpers::ActivationTypes::Atan:
            return std::make_shared<ov::op::v0::Atan>(in);
        case ov::helpers::ActivationTypes::Atanh:
            return std::make_shared<ov::op::v3::Atanh>(in);
        case ov::helpers::ActivationTypes::Cos:
            return std::make_shared<ov::op::v0::Cos>(in);
        case ov::helpers::ActivationTypes::Cosh:
            return std::make_shared<ov::op::v0::Cosh>(in);
        case ov::helpers::ActivationTypes::Floor:
            return std::make_shared<ov::op::v0::Floor>(in);
        case ov::helpers::ActivationTypes::Sin:
            return std::make_shared<ov::op::v0::Sin>(in);
        case ov::helpers::ActivationTypes::Sinh:
            return std::make_shared<ov::op::v0::Sinh>(in);
        case ov::helpers::ActivationTypes::Sqrt:
            return std::make_shared<ov::op::v0::Sqrt>(in);
        case ov::helpers::ActivationTypes::Tan:
            return std::make_shared<ov::op::v0::Tan>(in);
        case ov::helpers::ActivationTypes::Elu:
            return std::make_shared<ov::op::v0::Elu>(in, constantsValue[0]);
        case ov::helpers::ActivationTypes::Erf:
            return std::make_shared<ov::op::v0::Erf>(in);
        case ov::helpers::ActivationTypes::HardSigmoid: {
            auto hard_sigmoid_alpha = std::make_shared<ov::op::v0::Constant>(
                    type, inShape, constantsValue[0]);
            auto hard_sigmoid_beta = std::make_shared<ov::op::v0::Constant>(
                    type, inShape, constantsValue[1]);
            return std::make_shared<ov::op::v0::HardSigmoid>(in, hard_sigmoid_alpha, hard_sigmoid_beta);
        }
        case ov::helpers::ActivationTypes::Selu: {
            auto selu_alpha = std::make_shared<ov::op::v0::Constant>(
                    type, inShape, constantsValue[0]);
            auto selu_lambda = std::make_shared<ov::op::v0::Constant>(
                    type, inShape, constantsValue[1]);
            return std::make_shared<ov::op::v0::Selu>(in, selu_alpha, selu_lambda);
        }
        case ov::helpers::ActivationTypes::Ceiling:
            return std::make_shared<ov::op::v0::Ceiling>(in);
        case ov::helpers::ActivationTypes::PReLu: {
            auto negative_slope = std::make_shared<ov::op::v0::Constant>(
                    ov::element::f32,
                    inShape,
                    constantsValue);
            return std::make_shared<ov::op::v0::PRelu>(in, negative_slope);
        }
        case ov::helpers::ActivationTypes::Mish:
            return std::make_shared<ov::op::v4::Mish>(in);
        case ov::helpers::ActivationTypes::HSwish:
            return std::make_shared<ov::op::v4::HSwish>(in);
        case ov::helpers::ActivationTypes::SoftPlus:
            return std::make_shared<ov::op::v4::SoftPlus>(in);
        case ov::helpers::ActivationTypes::Swish: {
            auto beta = std::make_shared<ov::op::v0::Constant>(type, inShape, constantsValue[0]);
            return std::make_shared<ov::op::v4::Swish>(in, beta);
        }
        case ov::helpers::ActivationTypes::HSigmoid:
            return std::make_shared<ov::op::v5::HSigmoid>(in);
        case ov::helpers::ActivationTypes::RoundHalfToEven:
            return std::make_shared<ov::op::v5::Round>(in, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
        case ov::helpers::ActivationTypes::RoundHalfAwayFromZero:
            return std::make_shared<ov::op::v5::Round>(in, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
        case ov::helpers::ActivationTypes::GeluErf:
            return std::make_shared<ov::op::v7::Gelu>(in, ov::op::GeluApproximationMode::ERF);
        case ov::helpers::ActivationTypes::GeluTanh:
            return std::make_shared<ov::op::v7::Gelu>(in, ov::op::GeluApproximationMode::TANH);
        case ov::helpers::ActivationTypes::SoftSign:
            return std::make_shared<ov::op::v9::SoftSign>(in);
        default:
            throw std::runtime_error("Can't create layer for this activation type");
    }
}

std::shared_ptr<ov::Node> makeActivation(const ov::ParameterVector &parameters,
                                             const element::Type &type,
                                             ov::helpers::ActivationTypes activationType) {
    switch (activationType) {
        case ov::helpers::ActivationTypes::LeakyRelu:
            return std::make_shared<ov::op::v0::PRelu>(parameters[0], parameters[1]);
        case ov::helpers::ActivationTypes::HardSigmoid:
            return std::make_shared<ov::op::v0::HardSigmoid>(parameters[0], parameters[1], parameters[2]);
        case ov::helpers::ActivationTypes::Selu:
            return std::make_shared<ov::op::v0::Selu>(parameters[0], parameters[1], parameters[2]);
        case ov::helpers::ActivationTypes::PReLu:
            return std::make_shared<ov::op::v0::PRelu>(parameters[0], parameters[1]);
        default:
            throw std::runtime_error("It is impossible to create layer for this activation type with input as parameter");
    }
}

}  // namespace builder
}  // namespace ov
