// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/activation.hpp"

#include "openvino/op/abs.hpp"
#include "openvino/op/acos.hpp"
#include "openvino/op/acosh.hpp"
#include "openvino/op/asin.hpp"
#include "openvino/op/asinh.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/atanh.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cosh.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/hard_sigmoid.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/is_finite.hpp"
#include "openvino/op/is_inf.hpp"
#include "openvino/op/is_nan.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/mish.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/selu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/sinh.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/softsign.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tan.hpp"
#include "openvino/op/tanh.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_activation(const ov::Output<Node>& in,
                                          const element::Type& type,
                                          ov::test::utils::ActivationTypes activation_type,
                                          ov::Shape in_shape,
                                          std::vector<float> constants_value) {
    switch (activation_type) {
    case ov::test::utils::ActivationTypes::Sigmoid:
        return std::make_shared<ov::op::v0::Sigmoid>(in);
    case ov::test::utils::ActivationTypes::Tanh:
        return std::make_shared<ov::op::v0::Tanh>(in);
    case ov::test::utils::ActivationTypes::Relu:
        return std::make_shared<ov::op::v0::Relu>(in);
    case ov::test::utils::ActivationTypes::LeakyRelu: {
        auto leaky_slope = std::make_shared<ov::op::v0::Constant>(type, in_shape, constants_value);
        return std::make_shared<ov::op::v0::PRelu>(in, leaky_slope);
    }
    case ov::test::utils::ActivationTypes::Exp:
        return std::make_shared<ov::op::v0::Exp>(in);
    case ov::test::utils::ActivationTypes::Log:
        return std::make_shared<ov::op::v0::Log>(in);
    case ov::test::utils::ActivationTypes::Sign:
        return std::make_shared<ov::op::v0::Sign>(in);
    case ov::test::utils::ActivationTypes::Abs:
        return std::make_shared<ov::op::v0::Abs>(in);
    case ov::test::utils::ActivationTypes::Gelu:
        return std::make_shared<ov::op::v0::Gelu>(in);
    case ov::test::utils::ActivationTypes::Clamp:
        return std::make_shared<ov::op::v0::Clamp>(in, constants_value[0], constants_value[1]);
    case ov::test::utils::ActivationTypes::Negative:
        return std::make_shared<ov::op::v0::Negative>(in);
    case ov::test::utils::ActivationTypes::Acos:
        return std::make_shared<ov::op::v0::Acos>(in);
    case ov::test::utils::ActivationTypes::Acosh:
        return std::make_shared<ov::op::v3::Acosh>(in);
    case ov::test::utils::ActivationTypes::Asin:
        return std::make_shared<ov::op::v0::Asin>(in);
    case ov::test::utils::ActivationTypes::Asinh:
        return std::make_shared<ov::op::v3::Asinh>(in);
    case ov::test::utils::ActivationTypes::Atan:
        return std::make_shared<ov::op::v0::Atan>(in);
    case ov::test::utils::ActivationTypes::Atanh:
        return std::make_shared<ov::op::v3::Atanh>(in);
    case ov::test::utils::ActivationTypes::Cos:
        return std::make_shared<ov::op::v0::Cos>(in);
    case ov::test::utils::ActivationTypes::Cosh:
        return std::make_shared<ov::op::v0::Cosh>(in);
    case ov::test::utils::ActivationTypes::Floor:
        return std::make_shared<ov::op::v0::Floor>(in);
    case ov::test::utils::ActivationTypes::Sin:
        return std::make_shared<ov::op::v0::Sin>(in);
    case ov::test::utils::ActivationTypes::Sinh:
        return std::make_shared<ov::op::v0::Sinh>(in);
    case ov::test::utils::ActivationTypes::Sqrt:
        return std::make_shared<ov::op::v0::Sqrt>(in);
    case ov::test::utils::ActivationTypes::Tan:
        return std::make_shared<ov::op::v0::Tan>(in);
    case ov::test::utils::ActivationTypes::Elu:
        return std::make_shared<ov::op::v0::Elu>(in, constants_value[0]);
    case ov::test::utils::ActivationTypes::Erf:
        return std::make_shared<ov::op::v0::Erf>(in);
    case ov::test::utils::ActivationTypes::HardSigmoid: {
        auto hard_sigmoid_alpha = std::make_shared<ov::op::v0::Constant>(type, in_shape, constants_value[0]);
        auto hard_sigmoid_beta = std::make_shared<ov::op::v0::Constant>(type, in_shape, constants_value[1]);
        return std::make_shared<ov::op::v0::HardSigmoid>(in, hard_sigmoid_alpha, hard_sigmoid_beta);
    }
    case ov::test::utils::ActivationTypes::Selu: {
        auto selu_alpha = std::make_shared<ov::op::v0::Constant>(type, in_shape, constants_value[0]);
        auto selu_lambda = std::make_shared<ov::op::v0::Constant>(type, in_shape, constants_value[1]);
        return std::make_shared<ov::op::v0::Selu>(in, selu_alpha, selu_lambda);
    }
    case ov::test::utils::ActivationTypes::Ceiling:
        return std::make_shared<ov::op::v0::Ceiling>(in);
    case ov::test::utils::ActivationTypes::PReLu: {
        auto negative_slope = std::make_shared<ov::op::v0::Constant>(type, in_shape, constants_value);
        return std::make_shared<ov::op::v0::PRelu>(in, negative_slope);
    }
    case ov::test::utils::ActivationTypes::Mish:
        return std::make_shared<ov::op::v4::Mish>(in);
    case ov::test::utils::ActivationTypes::HSwish:
        return std::make_shared<ov::op::v4::HSwish>(in);
    case ov::test::utils::ActivationTypes::SoftPlus:
        return std::make_shared<ov::op::v4::SoftPlus>(in);
    case ov::test::utils::ActivationTypes::Swish: {
        auto beta = std::make_shared<ov::op::v0::Constant>(type, in_shape, constants_value[0]);
        return std::make_shared<ov::op::v4::Swish>(in, beta);
    }
    case ov::test::utils::ActivationTypes::HSigmoid:
        return std::make_shared<ov::op::v5::HSigmoid>(in);
    case ov::test::utils::ActivationTypes::RoundHalfToEven:
        return std::make_shared<ov::op::v5::Round>(in, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    case ov::test::utils::ActivationTypes::RoundHalfAwayFromZero:
        return std::make_shared<ov::op::v5::Round>(in, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    case ov::test::utils::ActivationTypes::GeluErf:
        return std::make_shared<ov::op::v7::Gelu>(in, ov::op::GeluApproximationMode::ERF);
    case ov::test::utils::ActivationTypes::GeluTanh:
        return std::make_shared<ov::op::v7::Gelu>(in, ov::op::GeluApproximationMode::TANH);
    case ov::test::utils::ActivationTypes::SoftSign:
        return std::make_shared<ov::op::v9::SoftSign>(in);
    case ov::test::utils::ActivationTypes::IsFinite:
        return std::make_shared<ov::op::v10::IsFinite>(in);
    case ov::test::utils::ActivationTypes::IsInf:
        return std::make_shared<ov::op::v10::IsInf>(in);
    case ov::test::utils::ActivationTypes::IsNaN:
        return std::make_shared<ov::op::v10::IsNaN>(in);
    case ov::test::utils::ActivationTypes::LogicalNot:
        return std::make_shared<ov::op::v1::LogicalNot>(in);
    default:
        OPENVINO_THROW("Can't create layer for this activation type");
    }
}

std::shared_ptr<ov::Node> make_activation(const ov::ParameterVector& parameters,
                                          const element::Type& type,
                                          ov::test::utils::ActivationTypes activation_type) {
    switch (activation_type) {
    case ov::test::utils::ActivationTypes::LeakyRelu:
        return std::make_shared<ov::op::v0::PRelu>(parameters[0], parameters[1]);
    case ov::test::utils::ActivationTypes::HardSigmoid:
        return std::make_shared<ov::op::v0::HardSigmoid>(parameters[0], parameters[1], parameters[2]);
    case ov::test::utils::ActivationTypes::Selu:
        return std::make_shared<ov::op::v0::Selu>(parameters[0], parameters[1], parameters[2]);
    case ov::test::utils::ActivationTypes::PReLu:
        return std::make_shared<ov::op::v0::PRelu>(parameters[0], parameters[1]);
    default:
        OPENVINO_THROW("It is impossible to create layer for this activation type with input as parameter");
    }
}
}  // namespace utils
}  // namespace test
}  // namespace ov
