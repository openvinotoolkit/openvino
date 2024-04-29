// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "transformations/utils/utils.hpp"

#include "openvino/op/tanh.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/asin.hpp"
#include "openvino/op/asinh.hpp"
#include "openvino/op/acos.hpp"
#include "openvino/op/acosh.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/atanh.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/hard_sigmoid.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/selu.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/tan.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/sinh.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cosh.hpp"
#include "openvino/op/softsign.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/mish.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/round.hpp"

#include "intel_gpu/primitives/activation.hpp"

namespace ov {
namespace intel_gpu {

void CreateUnaryEltwiseOp(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op,
                          cldnn::activation_func func, cldnn::activation_additional_params params) {
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);
    auto activationPrimitive = cldnn::activation(layerName, inputs[0], func, params);
    p.add_primitive(*op, activationPrimitive);
}

static void CreateTanhOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Tanh>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::hyperbolic_tan, {});
}

static void CreateEluOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Elu>& op) {
    auto alpha = static_cast<float>(op->get_alpha());
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::elu, {alpha});
}

static void CreateSigmoidOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Sigmoid>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::logistic, {});
}

static void CreateReluOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Relu>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::relu, {});
}

static void CreatePReluOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::PRelu>& op) {
    validate_inputs_count(op, {2});

    auto slope_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    auto slope_shape = op->get_input_partial_shape(1);
    auto out_shape = op->get_output_partial_shape(0);

    if (slope_node && ov::shape_size(slope_shape.to_shape()) == 1) {
        float slope;
        OPENVINO_ASSERT(ov::op::util::get_single_value(slope_node, slope),
                        "[GPU] Unsupported parameter size in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
        CreateUnaryEltwiseOp(p, op, cldnn::activation_func::relu_negative_slope, {slope});
    } else if (out_shape.size() >= 2) {
        auto inputs = p.GetInputInfo(op);
        std::string layerName = layer_type_name_ID(op);
        auto activationPrimitive = cldnn::activation(layerName,
                                                     inputs[0],
                                                     inputs[1].pid,
                                                     cldnn::activation_func::relu_negative_slope);
        p.add_primitive(*op, activationPrimitive);
    }
}

static void CreateClampOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Clamp>& op) {
    double min = op->get_min();
    double max = op->get_max();
    if (op->get_output_element_type(0) == ov::element::i32) {
        // Currently jitter saves all compile time constant as floats
        // and we have a code like that: (int)(as_float(0x4f000000))
        // So values in range (2147483583.0, 2147483647.0] are converted to  2147483648.0 due to fp32 representation error
        // and then conversion back to int32 returns -2147483648 due to overflow
        // So to avoid this issue we use largest representable value which doesn't cause overflow
        // TODO: Consider improving jitter to operate with int types directly
        max = std::min<double>(2147483583.0, max);
    }
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::clamp, {static_cast<float>(min), static_cast<float>(max)});
}

static void CreateExpOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Exp>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::exp, {});
}

static void CreateLogicalNotOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::LogicalNot>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::negation, {});
}

static void CreateAsinOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Asin>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::asin, {});
}

static void CreateAsinhOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v3::Asinh>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::asinh, {});
}

static void CreateAcosOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Acos>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::acos, {});
}

static void CreateAcoshOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v3::Acosh>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::acosh, {});
}

static void CreateAtanOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Atan>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::atan, {});
}

static void CreateAtanhOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v3::Atanh>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::atanh, {});
}

static void CreateAbsOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Abs>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::abs, {});
}

static void CreateFloorOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Floor>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::floor, {});
}

static void CreateCeilingOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Ceiling>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::ceil, {});
}

static void CreateSqrtOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Sqrt>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::sqrt, {});
}

static void CreateErfOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Erf>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::erf, {});
}

static void CreateHardSigmoidOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::HardSigmoid>& op) {
    validate_inputs_count(op, {3});
    auto alpha_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    auto beta_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
    if (!alpha_node || !beta_node) {
        OPENVINO_THROW("[GPU] Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
    }

    if (ov::shape_size(alpha_node->get_output_shape(0)) == 1 &&
        ov::shape_size(beta_node->get_output_shape(0)) == 1)  {
        float alpha, beta;
        if (!ov::op::util::get_single_value(alpha_node, alpha) || !ov::op::util::get_single_value(beta_node, beta)) {
            OPENVINO_THROW("Unsupported parameter size in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
        }
        CreateUnaryEltwiseOp(p, op, cldnn::activation_func::hard_sigmoid, {alpha, beta});
    }
}

static void CreateLogOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Log>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::log, {});
}

static void CreateNegativeOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Negative>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::negative, {});
}

static void CreateSeluOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Selu>& op) {
    validate_inputs_count(op, {3});
    auto alpha_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    auto lambda_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
    if (!alpha_node || !lambda_node) {
        OPENVINO_THROW("Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
    }

    if (ov::shape_size(alpha_node->get_output_shape(0)) == 1 &&
        ov::shape_size(lambda_node->get_output_shape(0)) == 1)  {
        float alpha, lambda;
        if (!ov::op::util::get_single_value(alpha_node, alpha) || !ov::op::util::get_single_value(lambda_node, lambda)) {
            OPENVINO_THROW("Unsupported parameter size in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
        }
        CreateUnaryEltwiseOp(p, op, cldnn::activation_func::selu, {alpha, lambda});
    } else {
        OPENVINO_THROW("Unsupported shapes of parameter nodes in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
    }
}

static void CreateSoftPlusOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v4::SoftPlus>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::softplus, {});
}

static void CreateTanOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Tan>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::tan, {});
}

static void CreateSinOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Sin>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::sin, {});
}

static void CreateSinhOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Sinh>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::sinh, {});
}

static void CreateCosOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Cos>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::cos, {});
}

static void CreateCoshOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Cosh>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::cosh, {});
}

static void CreateSwishOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v4::Swish>& op) {
    validate_inputs_count(op, {1, 2});
    if (op->get_input_size() == 2) {
        auto beta_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
        if (beta_node) {
            if (ov::shape_size(beta_node->get_output_shape(0)) == 1) {
                float beta;
                if (!ov::op::util::get_single_value(beta_node, beta)) {
                    OPENVINO_THROW("Unsupported parameter size in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
                }
                CreateUnaryEltwiseOp(p, op, cldnn::activation_func::swish, {beta});
            } else {
                OPENVINO_THROW("Unsupported parameter size in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
            }
        } else {
            OPENVINO_THROW("Unsupported parameter type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
        }
    } else {
        CreateUnaryEltwiseOp(p, op, cldnn::activation_func::swish, {1.0f});
    }
}

static void CreateHSwishOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v4::HSwish>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::hswish, {});
}

static void CreateMishOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v4::Mish>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::mish, {});
}

static void CreateGeluOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v7::Gelu>& op) {
    cldnn::activation_func activationFunc =
            op->get_approximation_mode() == op::GeluApproximationMode::ERF ? cldnn::activation_func::gelu
                                                                           : cldnn::activation_func::gelu_tanh;
    CreateUnaryEltwiseOp(p, op, activationFunc, {});
}

static void CreateSoftSignOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v9::SoftSign>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::softsign, {});
}

static void CreateGeluOp(ProgramBuilder &p, const std::shared_ptr<ov::op::v0::Gelu>& op) {
    CreateUnaryEltwiseOp(p, op,  cldnn::activation_func::gelu, {});
}

static void CreateSignOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Sign>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::sign, {});
}

static void CreateHSigmoidOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v5::HSigmoid>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::hsigmoid, {});
}

static void CreateRoundOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v5::Round>& op) {
    auto func = cldnn::activation_func::none;
    switch (op->get_mode()) {
        case ov::op::v5::Round::RoundMode::HALF_TO_EVEN : func = cldnn::activation_func::round_half_to_even; break;
        case ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO : func = cldnn::activation_func::round_half_away_from_zero; break;
        default: OPENVINO_THROW("Unsupported round mode in ", op->get_friendly_name(), ": ", static_cast<int>(op->get_mode()));
    }
    CreateUnaryEltwiseOp(p, op, func, {});
}

REGISTER_FACTORY_IMPL(v0, Tanh);
REGISTER_FACTORY_IMPL(v0, Elu);
REGISTER_FACTORY_IMPL(v0, Sigmoid);
REGISTER_FACTORY_IMPL(v0, Relu);
REGISTER_FACTORY_IMPL(v0, PRelu);
REGISTER_FACTORY_IMPL(v0, Clamp);
REGISTER_FACTORY_IMPL(v0, Exp);
REGISTER_FACTORY_IMPL(v1, LogicalNot);
REGISTER_FACTORY_IMPL(v0, Asin);
REGISTER_FACTORY_IMPL(v3, Asinh);
REGISTER_FACTORY_IMPL(v0, Acos);
REGISTER_FACTORY_IMPL(v3, Acosh);
REGISTER_FACTORY_IMPL(v0, Atan);
REGISTER_FACTORY_IMPL(v3, Atanh);
REGISTER_FACTORY_IMPL(v0, Abs);
REGISTER_FACTORY_IMPL(v0, Floor);
REGISTER_FACTORY_IMPL(v0, Ceiling);
REGISTER_FACTORY_IMPL(v0, Sqrt);
REGISTER_FACTORY_IMPL(v0, Erf);
REGISTER_FACTORY_IMPL(v0, HardSigmoid);
REGISTER_FACTORY_IMPL(v0, Log);
REGISTER_FACTORY_IMPL(v0, Negative);
REGISTER_FACTORY_IMPL(v0, Selu);
REGISTER_FACTORY_IMPL(v4, SoftPlus);
REGISTER_FACTORY_IMPL(v0, Tan);
REGISTER_FACTORY_IMPL(v0, Sin);
REGISTER_FACTORY_IMPL(v0, Sinh);
REGISTER_FACTORY_IMPL(v0, Cos);
REGISTER_FACTORY_IMPL(v0, Cosh);
REGISTER_FACTORY_IMPL(v4, Swish);
REGISTER_FACTORY_IMPL(v4, HSwish);
REGISTER_FACTORY_IMPL(v4, Mish);
REGISTER_FACTORY_IMPL(v0, Gelu);
REGISTER_FACTORY_IMPL(v7, Gelu);
REGISTER_FACTORY_IMPL(v0, Sign);
REGISTER_FACTORY_IMPL(v5, HSigmoid);
REGISTER_FACTORY_IMPL(v5, Round);
REGISTER_FACTORY_IMPL(v9, SoftSign);

}  // namespace intel_gpu
}  // namespace ov
