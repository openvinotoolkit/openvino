// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "transformations/utils/utils.hpp"

#include "ngraph/op/tanh.hpp"
#include "ngraph/op/elu.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/prelu.hpp"
#include "ngraph/op/clamp.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/asinh.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/acosh.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/atanh.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/erf.hpp"
#include "ngraph/op/hard_sigmoid.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/selu.hpp"
#include "ngraph/op/softplus.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/swish.hpp"
#include "ngraph/op/hswish.hpp"
#include "ngraph/op/mish.hpp"
#include "ngraph/op/gelu.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/hsigmoid.hpp"
#include "ngraph/op/round.hpp"

#include "intel_gpu/primitives/activation.hpp"

namespace ov {
namespace intel_gpu {

void CreateUnaryEltwiseOp(Program& p, const std::shared_ptr<ngraph::Node>& op,
                          cldnn::activation_func func, cldnn::activation_additional_params params) {
    auto inputs = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);
    auto activationPrimitive = cldnn::activation(layerName, inputs[0], func, params, op->get_friendly_name());
    p.AddPrimitive(activationPrimitive);
    p.AddPrimitiveToProfiler(op);
}

static void CreateTanhOp(Program& p, const std::shared_ptr<ngraph::op::v0::Tanh>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::hyperbolic_tan, {});
}

static void CreateEluOp(Program& p, const std::shared_ptr<ngraph::op::v0::Elu>& op) {
    auto alpha = static_cast<float>(op->get_alpha());
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::elu, {alpha});
}

static void CreateSigmoidOp(Program& p, const std::shared_ptr<ngraph::op::v0::Sigmoid>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::logistic, {});
}

static void CreateReluOp(Program& p, const std::shared_ptr<ngraph::op::v0::Relu>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::relu, {});
}

static void CreatePReluOp(Program& p, const std::shared_ptr<ngraph::op::v0::PRelu>& op) {
    p.ValidateInputs(op, {2});

    auto slope_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    auto slope_shape = op->get_input_shape(1);
    auto out_shape = op->get_output_shape(0);

    if (slope_node && ngraph::shape_size(slope_shape) == 1) {
        float slope;
        if (!ngraph::op::util::get_single_value(slope_node, slope))
            IE_THROW() << "Unsupported parameter size in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        CreateUnaryEltwiseOp(p, op, cldnn::activation_func::relu_negative_slope, {slope});
    } else if (out_shape.size() >= 2) {
        auto inputs = p.GetInputPrimitiveIDs(op);
        std::string layerName = layer_type_name_ID(op);
        auto activationPrimitive = cldnn::activation(layerName,
                                                     inputs[0],
                                                     inputs[1],
                                                     cldnn::activation_func::relu_negative_slope,
                                                     op->get_friendly_name());
        p.AddPrimitive(activationPrimitive);
        p.AddPrimitiveToProfiler(op);
    }
}

static void CreateClampOp(Program& p, const std::shared_ptr<ngraph::op::v0::Clamp>& op) {
    float min = static_cast<float>(op->get_min());
    float max = static_cast<float>(op->get_max());
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::clamp, {min, max});
}

static void CreateExpOp(Program& p, const std::shared_ptr<ngraph::op::v0::Exp>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::exp, {});
}

static void CreateLogicalNotOp(Program& p, const std::shared_ptr<ngraph::op::v1::LogicalNot>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::negation, {});
}

static void CreateAsinOp(Program& p, const std::shared_ptr<ngraph::op::v0::Asin>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::asin, {});
}

static void CreateAsinhOp(Program& p, const std::shared_ptr<ngraph::op::v3::Asinh>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::asinh, {});
}

static void CreateAcosOp(Program& p, const std::shared_ptr<ngraph::op::v0::Acos>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::acos, {});
}

static void CreateAcoshOp(Program& p, const std::shared_ptr<ngraph::op::v3::Acosh>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::acosh, {});
}

static void CreateAtanOp(Program& p, const std::shared_ptr<ngraph::op::v0::Atan>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::atan, {});
}

static void CreateAtanhOp(Program& p, const std::shared_ptr<ngraph::op::v3::Atanh>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::atanh, {});
}

static void CreateAbsOp(Program& p, const std::shared_ptr<ngraph::op::v0::Abs>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::abs, {});
}

static void CreateFloorOp(Program& p, const std::shared_ptr<ngraph::op::v0::Floor>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::floor, {});
}

static void CreateCeilingOp(Program& p, const std::shared_ptr<ngraph::op::v0::Ceiling>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::ceil, {});
}

static void CreateSqrtOp(Program& p, const std::shared_ptr<ngraph::op::v0::Sqrt>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::sqrt, {});
}

static void CreateErfOp(Program& p, const std::shared_ptr<ngraph::op::v0::Erf>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::erf, {});
}

static void CreateHardSigmoidOp(Program& p, const std::shared_ptr<ngraph::op::v0::HardSigmoid>& op) {
    p.ValidateInputs(op, {3});
    auto alpha_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    auto beta_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(2));
    if (!alpha_node || !beta_node) {
        IE_THROW() << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
    }

    if (ngraph::shape_size(alpha_node->get_output_shape(0)) == 1 &&
        ngraph::shape_size(beta_node->get_output_shape(0)) == 1)  {
        float alpha, beta;
        if (!ngraph::op::util::get_single_value(alpha_node, alpha) || !ngraph::op::util::get_single_value(beta_node, beta)) {
            IE_THROW() << "Unsupported parameter size in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        }
        CreateUnaryEltwiseOp(p, op, cldnn::activation_func::hard_sigmoid, {alpha, beta});
    }
}

static void CreateLogOp(Program& p, const std::shared_ptr<ngraph::op::v0::Log>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::log, {});
}

static void CreateNegativeOp(Program& p, const std::shared_ptr<ngraph::op::v0::Negative>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::negative, {});
}

static void CreateSeluOp(Program& p, const std::shared_ptr<ngraph::op::v0::Selu>& op) {
    p.ValidateInputs(op, {3});
    auto alpha_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    auto lambda_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(2));
    if (!alpha_node || !lambda_node) {
        IE_THROW() << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
    }

    if (ngraph::shape_size(alpha_node->get_output_shape(0)) == 1 &&
        ngraph::shape_size(lambda_node->get_output_shape(0)) == 1)  {
        float alpha, lambda;
        if (!ngraph::op::util::get_single_value(alpha_node, alpha) || !ngraph::op::util::get_single_value(lambda_node, lambda)) {
            IE_THROW() << "Unsupported parameter size in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        }
        CreateUnaryEltwiseOp(p, op, cldnn::activation_func::selu, {alpha, lambda});
    } else {
        IE_THROW() << "Unsupported shapes of parameter nodes in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
    }
}

static void CreateSoftPlusOp(Program& p, const std::shared_ptr<ngraph::op::v4::SoftPlus>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::softplus, {});
}

static void CreateTanOp(Program& p, const std::shared_ptr<ngraph::op::v0::Tan>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::tan, {});
}

static void CreateSinOp(Program& p, const std::shared_ptr<ngraph::op::v0::Sin>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::sin, {});
}

static void CreateSinhOp(Program& p, const std::shared_ptr<ngraph::op::v0::Sinh>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::sinh, {});
}

static void CreateCosOp(Program& p, const std::shared_ptr<ngraph::op::v0::Cos>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::cos, {});
}

static void CreateCoshOp(Program& p, const std::shared_ptr<ngraph::op::v0::Cosh>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::cosh, {});
}

static void CreateSwishOp(Program& p, const std::shared_ptr<ngraph::op::v4::Swish>& op) {
    p.ValidateInputs(op, {1, 2});
    if (op->get_input_size() == 2) {
        auto beta_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1));
        if (beta_node) {
            if (ngraph::shape_size(beta_node->get_output_shape(0)) == 1) {
                float beta;
                if (!ngraph::op::util::get_single_value(beta_node, beta)) {
                    IE_THROW() << "Unsupported parameter size in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
                }
                CreateUnaryEltwiseOp(p, op, cldnn::activation_func::swish, {beta});
            } else {
                IE_THROW() << "Unsupported parameter size in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
            }
        } else {
            IE_THROW() << "Unsupported parameter type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        }
    } else {
        CreateUnaryEltwiseOp(p, op, cldnn::activation_func::swish, {1.0f});
    }
}

static void CreateHSwishOp(Program& p, const std::shared_ptr<ngraph::op::v4::HSwish>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::hswish, {});
}

static void CreateMishOp(Program& p, const std::shared_ptr<ngraph::op::v4::Mish>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::mish, {});
}

static void CreateGeluOp(Program& p, const std::shared_ptr<ngraph::op::v7::Gelu>& op) {
    cldnn::activation_func activationFunc =
            op->get_approximation_mode() == op::GeluApproximationMode::ERF ? cldnn::activation_func::gelu
                                                                           : cldnn::activation_func::gelu_tanh;
    CreateUnaryEltwiseOp(p, op, activationFunc, {});
}

static void CreateSoftSignOp(Program& p, const std::shared_ptr<ngraph::op::v9::SoftSign>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::softsign, {});
}

static void CreateGeluOp(Program &p, const std::shared_ptr<ngraph::op::v0::Gelu>& op) {
    CreateUnaryEltwiseOp(p, op,  cldnn::activation_func::gelu, {});
}

static void CreateSignOp(Program& p, const std::shared_ptr<ngraph::op::v0::Sign>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::sign, {});
}

static void CreateHSigmoidOp(Program& p, const std::shared_ptr<ngraph::op::v5::HSigmoid>& op) {
    CreateUnaryEltwiseOp(p, op, cldnn::activation_func::hsigmoid, {});
}

static void CreateRoundOp(Program& p, const std::shared_ptr<ngraph::op::v5::Round>& op) {
    auto func = cldnn::activation_func::none;
    switch (op->get_mode()) {
        case ngraph::op::v5::Round::RoundMode::HALF_TO_EVEN : func = cldnn::activation_func::round_half_to_even; break;
        case ngraph::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO : func = cldnn::activation_func::round_half_away_from_zero; break;
        default: IE_THROW() << "Unsupported round mode in " << op->get_friendly_name() << ": " << static_cast<int>(op->get_mode());
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
