// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise.h"

#include <oneapi/dnnl/dnnl_types.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "config.h"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "dnnl_postops_composer_legacy.h"
#include "edge.h"
#include "graph_context.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/eltwise_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/jit/eltwise.h"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/kernels/jit_eltwise_common.hpp"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/enum_names.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_left_shift.hpp"
#include "openvino/op/bitwise_not.hpp"
#include "openvino/op/bitwise_or.hpp"
#include "openvino/op/bitwise_right_shift.hpp"
#include "openvino/op/bitwise_xor.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/is_finite.hpp"
#include "openvino/op/is_inf.hpp"
#include "openvino/op/is_nan.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/mish.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/softsign.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "post_ops.hpp"
#include "shape_inference/custom/eltwise.hpp"
#include "transformations/cpu_opset/common/op/leaky_relu.hpp"
#include "transformations/cpu_opset/common/op/power_static.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"
#include "utils/general_utils.h"
#include "utils/ngraph_utils.hpp"

namespace ov::intel_cpu::node {

EltwiseBroadcastingPolicy Eltwise::determineBroadcastingPolicy(const std::shared_ptr<ov::Node>& op) {
    const auto const1 = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(0));
    const auto const2 = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    int constPort = -1;
    if (const2) {
        constPort = 1;
    } else if (const1) {
        constPort = 0;
    } else {
        return EltwiseBroadcastingPolicy::Undefined;
    }

    auto const_shape = op->get_input_shape(constPort);
    if (ov::shape_size(const_shape) == 1) {
        return EltwiseBroadcastingPolicy::PerTensor;
    }
    return EltwiseBroadcastingPolicy::PerChannel;
}

const std::map<const ov::DiscreteTypeInfo, Eltwise::Initializer>& Eltwise::getInitializers() {
    static const std::map<const ov::DiscreteTypeInfo, Eltwise::Initializer> initializers{
        {ov::op::v1::Add::get_type_info_static(),
         [](const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseAdd;
             node.m_attrs.broadcastingPolicy = determineBroadcastingPolicy(op);
         }},
        {ov::op::v1::Subtract::get_type_info_static(),
         [](const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseSubtract;
             node.m_attrs.broadcastingPolicy = determineBroadcastingPolicy(op);
         }},
        {ov::op::v1::Multiply::get_type_info_static(),
         [](const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseMultiply;
             node.m_attrs.broadcastingPolicy = determineBroadcastingPolicy(op);
         }},
        {ov::op::v1::Divide::get_type_info_static(),
         [](const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseDivide;
             node.m_attrs.broadcastingPolicy = determineBroadcastingPolicy(op);
         }},
        {ov::op::v0::SquaredDifference::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseSquaredDifference;
         }},
        {ov::op::v1::Maximum::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseMaximum;
         }},
        {ov::op::v1::Minimum::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseMinimum;
         }},
        {ov::op::v1::Mod::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseMod;
         }},
        {ov::op::v0::Ceiling::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseCeiling;
         }},
        {ov::op::v0::Negative::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseNegative;
         }},
        {ov::op::v0::Floor::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseFloor;
         }},
        {ov::op::v1::FloorMod::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseFloorMod;
         }},
        {ov::op::v1::Power::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwisePowerDynamic;
         }},
        {PowerStaticNode::get_type_info_static(),
         [](const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             auto powerStatic = getNgraphOpAs<PowerStaticNode>(op);
             node.algorithm = Algorithm::EltwisePowerStatic;
             node.m_attrs.data.alpha = powerStatic->get_power();
             node.m_attrs.data.beta = powerStatic->get_scale();
             node.m_attrs.data.gamma = powerStatic->get_shift();
             node.m_attrs.broadcastingPolicy = EltwiseBroadcastingPolicy::PerTensor;
         }},
        {ov::op::v1::Equal::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseEqual;
         }},
        {ov::op::v1::NotEqual::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseNotEqual;
         }},
        {ov::op::v10::IsFinite::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseIsFinite;
         }},
        {ov::op::v10::IsInf::get_type_info_static(),
         [](const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseIsInf;
             const auto& attributes = ov::as_type_ptr<ov::op::v10::IsInf>(op)->get_attributes();
             node.m_attrs.data.alpha = static_cast<float>(attributes.detect_negative);
             node.m_attrs.data.beta = static_cast<float>(attributes.detect_positive);
         }},
        {ov::op::v10::IsNaN::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseIsNaN;
         }},
        {ov::op::v1::Greater::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseGreater;
         }},
        {ov::op::v1::GreaterEqual::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseGreaterEqual;
         }},
        {ov::op::v1::Less::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseLess;
         }},
        {ov::op::v1::LessEqual::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseLessEqual;
         }},
        {ov::op::v1::LogicalAnd::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseLogicalAnd;
         }},
        {ov::op::v1::LogicalOr::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseLogicalOr;
         }},
        {ov::op::v1::LogicalXor::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseLogicalXor;
         }},
        {ov::op::v1::LogicalNot::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseLogicalNot;
         }},
        {ov::op::v0::Relu::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseRelu;
             node.m_attrs.data.onednnAlgorithm = dnnl::algorithm::eltwise_relu;
         }},
        {LeakyReluNode::get_type_info_static(),
         [](const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             auto leakyRelu = getNgraphOpAs<LeakyReluNode>(op);
             node.algorithm = Algorithm::EltwiseRelu;
             node.m_attrs.data.onednnAlgorithm = dnnl::algorithm::eltwise_relu;
             node.m_attrs.data.alpha = leakyRelu->get_slope();
             node.m_attrs.data.beta = 0.0F;
         }},
        {ov::op::v0::Gelu::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseGeluErf;
             node.m_attrs.data.onednnAlgorithm = dnnl::algorithm::eltwise_gelu_erf;
         }},
        {ov::op::v7::Gelu::get_type_info_static(),
         [](const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             auto gelu = getNgraphOpAs<ov::op::v7::Gelu>(op);
             ov::op::GeluApproximationMode approximationMode = gelu->get_approximation_mode();
             if (approximationMode == ov::op::GeluApproximationMode::ERF) {
                 node.algorithm = Algorithm::EltwiseGeluErf;
                 node.m_attrs.data.onednnAlgorithm = dnnl::algorithm::eltwise_gelu_erf;
             } else if (approximationMode == ov::op::GeluApproximationMode::TANH) {
                 node.algorithm = Algorithm::EltwiseGeluTanh;
                 node.m_attrs.data.onednnAlgorithm = dnnl::algorithm::eltwise_gelu_tanh;
             } else {
                 OPENVINO_THROW_NOT_IMPLEMENTED(
                     "CPU Eltwise node doesn't support ngraph operation Gelu with approximation mode: ",
                     approximationMode);
             }
         }},
        {ov::op::v0::Elu::get_type_info_static(),
         [](const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             auto eluOp = getNgraphOpAs<ov::op::v0::Elu>(op);
             node.m_attrs.data.alpha = static_cast<float>(eluOp->get_alpha());
             node.algorithm = Algorithm::EltwiseElu;
             node.m_attrs.data.onednnAlgorithm = dnnl::algorithm::eltwise_elu;
         }},
        {ov::op::v0::Tanh::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseTanh;
             node.m_attrs.data.onednnAlgorithm = dnnl::algorithm::eltwise_tanh;
         }},
        {ov::op::v0::Sigmoid::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseSigmoid;
             node.m_attrs.data.onednnAlgorithm = dnnl::algorithm::eltwise_logistic;
         }},
        {ov::op::v0::Abs::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseAbs;
             node.m_attrs.data.onednnAlgorithm = dnnl::algorithm::eltwise_abs;
         }},
        {ov::op::v0::Sqrt::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseSqrt;
             node.m_attrs.data.onednnAlgorithm = dnnl::algorithm::eltwise_sqrt;
         }},
        {ov::op::v0::Clamp::get_type_info_static(),
         [](const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             auto clampOp = getNgraphOpAs<ov::op::v0::Clamp>(op);
             auto alpha_ = static_cast<float>(clampOp->get_min());
             auto beta_ = static_cast<float>(clampOp->get_max());
             if (clampOp->get_input_element_type(0).is_integral_number()) {
                 // according to spec, when Clamp has integer element type, min and max mist be converted to
                 // integer
                 alpha_ = std::ceil(alpha_);
                 beta_ = std::floor(beta_);
             }
             node.m_attrs.data.alpha = alpha_;
             node.m_attrs.data.beta = beta_;
             node.algorithm = Algorithm::EltwiseClamp;
             node.m_attrs.data.onednnAlgorithm = dnnl::algorithm::eltwise_clip;
         }},
        {ov::op::v0::Exp::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseExp;
         }},
        {SwishNode::get_type_info_static(),
         [](const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             auto swishOp = getNgraphOpAs<SwishNode>(op);
             node.algorithm = Algorithm::EltwiseSwish;
             node.m_attrs.data.onednnAlgorithm = dnnl::algorithm::eltwise_swish;
             node.m_attrs.data.alpha = swishOp->get_alpha();
         }},
        {ov::op::v4::HSwish::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             // since v3.0 version, oneDNN has flexible implementation of hardswish, ov still uses the one with
             // hardcoded alpha and beta
             node.m_attrs.data.alpha = 1.F / 6.F;
             node.m_attrs.data.beta = 0.5F;
             node.algorithm = Algorithm::EltwiseHswish;
             node.m_attrs.data.onednnAlgorithm = dnnl::algorithm::eltwise_hardswish;
         }},
        {ov::op::v4::Mish::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseMish;
             node.m_attrs.data.onednnAlgorithm = dnnl::algorithm::eltwise_mish;
         }},
        {ov::op::v5::HSigmoid::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseHsigmoid;
             node.m_attrs.data.onednnAlgorithm = dnnl::algorithm::eltwise_hsigmoid;
         }},
        {ov::op::v5::Round::get_type_info_static(),
         [](const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             auto roundOp = getNgraphOpAs<ov::op::v5::Round>(op);

             switch (roundOp->get_mode()) {
             case ov::op::v5::Round::RoundMode::HALF_TO_EVEN:
                 node.algorithm = Algorithm::EltwiseRoundHalfToEven;
                 node.m_attrs.data.onednnAlgorithm = dnnl::algorithm::eltwise_round_half_to_even;
                 break;
             case ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO:
                 node.algorithm = Algorithm::EltwiseRoundHalfAwayFromZero;
                 node.m_attrs.data.onednnAlgorithm = dnnl::algorithm::eltwise_round_half_away_from_zero;
                 break;
             }
         }},
        {ov::op::v0::PRelu::get_type_info_static(),
         [](const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwisePrelu;
             node.m_attrs.broadcastingPolicy = determineBroadcastingPolicy(op);
         }},
        {ov::op::v0::Erf::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseErf;
         }},
        {ov::op::v4::SoftPlus::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseSoftRelu;
             node.m_attrs.data.alpha = 1.F;
             node.m_attrs.data.onednnAlgorithm = dnnl::algorithm::eltwise_soft_relu;
         }},
        {ov::op::v9::SoftSign::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseSoftSign;
         }},
        {ov::op::v1::Select::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseSelect;
         }},
        {ov::op::v0::Log::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseLog;
         }},
        {op::v13::BitwiseAnd::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseBitwiseAnd;
         }},
        {op::v13::BitwiseNot::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseBitwiseNot;
         }},
        {op::v13::BitwiseOr::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseBitwiseOr;
         }},
        {op::v13::BitwiseXor::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseBitwiseXor;
         }},
        {op::v15::BitwiseLeftShift::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseBitwiseLeftShift;
         }},
        {op::v15::BitwiseRightShift::get_type_info_static(),
         []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Eltwise& node) {
             node.algorithm = Algorithm::EltwiseBitwiseRightShift;
         }},
    };
    return initializers;
}

bool Eltwise::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (getInitializers().find(op->get_type_info()) == getInitializers().end()) {
            errorMessage = "Doesn't support Eltwise algorithm: " + std::string(op->get_type_name());
            return false;
        }
        if (const auto binOp = ov::as_type_ptr<const ov::op::util::BinaryElementwiseArithmetic>(op)) {
            if (binOp->get_autob().m_type != ov::op::AutoBroadcastType::NONE &&
                binOp->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
                errorMessage = "Doesn't support broadcast type: " + ov::as_string(binOp->get_autob().m_type);
                return false;
            }
        }
        if (const auto select = ov::as_type_ptr<const ov::op::v1::Select>(op)) {
            if (select->get_auto_broadcast().m_type != ov::op::AutoBroadcastType::NONE &&
                select->get_auto_broadcast().m_type != ov::op::AutoBroadcastType::NUMPY) {
                errorMessage = "Doesn't support broadcast type: " + ov::as_string(select->get_autob().m_type);
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

Eltwise::Eltwise(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, EltwiseShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    getInitializers().at(op->get_type_info())(op, *this);
    m_attrs.data.algo = getAlgorithm();
}

size_t Eltwise::getOpInputsNum() const {
    switch (getAlgorithm()) {
    case Algorithm::EltwiseIsFinite:
    case Algorithm::EltwiseIsInf:
    case Algorithm::EltwiseIsNaN:
    case Algorithm::EltwiseRelu:
    case Algorithm::EltwiseGeluErf:
    case Algorithm::EltwiseGeluTanh:
    case Algorithm::EltwiseCeiling:
    case Algorithm::EltwiseNegative:
    case Algorithm::EltwiseFloor:
    case Algorithm::EltwiseElu:
    case Algorithm::EltwiseTanh:
    case Algorithm::EltwiseSigmoid:
    case Algorithm::EltwiseAbs:
    case Algorithm::EltwiseSqrt:
    case Algorithm::EltwiseSoftRelu:
    case Algorithm::EltwiseExp:
    case Algorithm::EltwiseClamp:
    case Algorithm::EltwiseErf:
    case Algorithm::EltwiseLogicalNot:
    case Algorithm::EltwisePowerStatic:
    case Algorithm::EltwiseSwish:
    case Algorithm::EltwiseHswish:
    case Algorithm::EltwiseMish:
    case Algorithm::EltwiseHsigmoid:
    case Algorithm::EltwiseRoundHalfToEven:
    case Algorithm::EltwiseRoundHalfAwayFromZero:
    case Algorithm::EltwiseSoftSign:
    case Algorithm::EltwiseLog:
        return 1;
    case Algorithm::EltwiseAdd:
    case Algorithm::EltwiseSubtract:
    case Algorithm::EltwiseMultiply:
    case Algorithm::EltwiseDivide:
    case Algorithm::EltwiseFloorMod:
    case Algorithm::EltwiseMod:
    case Algorithm::EltwiseMaximum:
    case Algorithm::EltwiseMinimum:
    case Algorithm::EltwiseSquaredDifference:
    case Algorithm::EltwisePowerDynamic:
    case Algorithm::EltwiseEqual:
    case Algorithm::EltwiseNotEqual:
    case Algorithm::EltwiseGreater:
    case Algorithm::EltwiseGreaterEqual:
    case Algorithm::EltwiseLess:
    case Algorithm::EltwiseLessEqual:
    case Algorithm::EltwiseLogicalAnd:
    case Algorithm::EltwiseLogicalOr:
    case Algorithm::EltwiseLogicalXor:
    case Algorithm::EltwiseBitwiseAnd:
    case Algorithm::EltwiseBitwiseOr:
    case Algorithm::EltwiseBitwiseXor:
    case Algorithm::EltwiseBitwiseLeftShift:
    case Algorithm::EltwiseBitwiseRightShift:
        return 2;
    case Algorithm::EltwiseBitwiseNot:
        return 1;
    case Algorithm::EltwisePrelu:
        return 2;
    case Algorithm::EltwiseMulAdd:
    case Algorithm::EltwiseSelect:
        return 3;
    default:
        CPU_NODE_THROW("Unsupported operation.");
    }
}

bool Eltwise::isWithBroadcast() {
    const auto& oDims = getOutputShapeAtPort(0).getDims();
    for (size_t i = 0; i < inputShapes.size(); i++) {
        const auto& iDims = getInputShapeAtPort(i).getDims();
        if (!dimsEqualWeak(iDims, oDims)) {
            return true;
        }
    }
    return false;
}

void Eltwise::getSupportedDescriptors() {
    CPU_NODE_ASSERT(!getParentEdges().empty(), "Incorrect number of input edges");
    CPU_NODE_ASSERT(!getChildEdges().empty(), "Incorrect number of output edges");
}

static void sortSupportedPrimitiveDescriptors(std::vector<NodeDesc>& supportedPrimitiveDescriptors,
                                              const Config::ModelType modelType) {
    // sort supportedPrimitiveDescriptors by layout type
    static const std::vector<LayoutType> cnnLayoutPriority{LayoutType::nspc,
                                                           LayoutType::nCsp16c,
                                                           LayoutType::nCsp8c,
                                                           LayoutType::ncsp};

    static const std::vector<LayoutType> restLayoutPriority{LayoutType::ncsp,
                                                            LayoutType::nspc,
                                                            LayoutType::nCsp16c,
                                                            LayoutType::nCsp8c};

    auto getLayoutType = [](const MemoryDescPtr& memDesc) {
        if (memDesc->hasLayoutType(LayoutType::ncsp)) {
            return LayoutType::ncsp;
        }

        if (memDesc->hasLayoutType(LayoutType::nspc)) {
            return LayoutType::nspc;
        }

        if (memDesc->hasLayoutType(LayoutType::nCsp16c)) {
            return LayoutType::nCsp16c;
        }

        if (memDesc->hasLayoutType(LayoutType::nCsp8c)) {
            return LayoutType::nCsp8c;
        }

        return LayoutType::nspc;
    };

    std::sort(supportedPrimitiveDescriptors.begin(),
              supportedPrimitiveDescriptors.end(),
              [&](const NodeDesc& a, const NodeDesc& b) {
                  const auto aLayout = getLayoutType(a.getConfig().outConfs[0].getMemDesc());
                  const auto bLayout = getLayoutType(b.getConfig().outConfs[0].getMemDesc());
                  const auto& layoutPriority =
                      modelType == Config::ModelType::CNN ? cnnLayoutPriority : restLayoutPriority;
                  return std::find(layoutPriority.begin(), layoutPriority.end(), aLayout) <
                         std::find(layoutPriority.begin(), layoutPriority.end(), bLayout);
              });
}

void Eltwise::initSupportedPrimitiveDescriptors() {
    size_t expectedInputsNum = getOpInputsNum();
    for (auto& postOp : fusedWith) {
        if (const auto* eltwiseNode = dynamic_cast<const Eltwise*>(postOp.get())) {
            expectedInputsNum += eltwiseNode->getOpInputsNum() - 1;
        }
    }

    CPU_NODE_ASSERT(getParentEdges().size() <= MAX_ELTWISE_INPUTS,
                    "doesn't support more than ",
                    MAX_ELTWISE_INPUTS,
                    " inputs (actual = ",
                    getParentEdges().size(),
                    ")");

    CPU_NODE_ASSERT(expectedInputsNum == getParentEdges().size(),
                    "has invalid input number of inputs: expected = ",
                    expectedInputsNum,
                    " (actual = ",
                    getParentEdges().size(),
                    ")");

    // Initialize attributes
    m_attrs.data.algo = getAlgorithm();
    m_attrs.postOps = getPostOps(fusedWith, ov::element::dynamic);
    m_attrs.opsList = {getType()};
    // Create memory descriptors
    std::vector<MemoryDescPtr> srcDescs;
    // Select preferred layout for memory descriptors
    const auto preferredLayout =
        context->getConfig().modelType == Config::ModelType::CNN ? LayoutType::nspc : LayoutType::ncsp;

    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();

    // Create src memory descriptors
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto srcDesc = creatorsMap.at(preferredLayout)
                           ->createSharedDesc(getOriginalInputPrecisionAtPort(i), getInputShapeAtPort(i));
        srcDescs.push_back(srcDesc);
    }

    // Create dst memory descriptors
    const auto dstPrecision = !fusedWith.empty() ? fusedWith.back()->getOriginalOutputPrecisionAtPort(0)
                                                 : getOriginalOutputPrecisionAtPort(0);
    auto dstDesc = creatorsMap.at(preferredLayout)->createSharedDesc(dstPrecision, getOutputShapeAtPort(0));

    // Prepare memory descriptor arguments for a factory
    MemoryDescArgs descs;
    for (size_t i = 0; i < srcDescs.size(); i++) {
        descs[ARG_SRC + i] = srcDescs[i];
    }
    descs[ARG_DST] = dstDesc;

    auto executionContext = std::make_shared<ExecutorContext>(context, getImplPriority());
    m_factory = std::make_shared<ExecutorFactory<EltwiseAttrs>>(m_attrs, executionContext, descs, memoryFormatFilter);

    const std::vector<MemoryDescArgs> nodeDescriptorsList = m_factory->getProperMemoryDescriptors(descs);

    for (const auto& nodeDescriptors : nodeDescriptorsList) {
        NodeConfig nodeConfig;
        nodeConfig.inConfs.resize(srcDescs.size());

        const auto& outputDesc = nodeDescriptors.at(ARG_DST);
        const auto outputPrecision = outputDesc->getPrecision();

        for (size_t i = 0; i < srcDescs.size(); i++) {
            if (auto it = nodeDescriptors.find(ARG_SRC + i); it != nodeDescriptors.end()) {
                const auto& [_, desc] = *it;
                const int isInPlace =
                    i == 0 && !isDynamicNode() && canBeInPlace() && desc->getPrecision() == outputPrecision ? 0 : -1;
                const auto& srcShape = getInputShapeAtPort(i);
                const bool acceptAnyBatchStride = !isDynamicNode() && srcShape.getDims()[0] == 1;
                const BlockedMemoryDesc::CmpMask inputMask =
                    acceptAnyBatchStride ? BlockedMemoryDesc::EMPTY_MASK : BlockedMemoryDesc::SKIP_OFFSET_MASK;
                nodeConfig.inConfs[i] = {desc, inputMask, isInPlace};
            }
        }

        const auto& dstShape = getOutputShapeAtPort(0);
        const bool acceptAnyBatchStride = !isDynamicNode() && dstShape.getDims()[0] == 1;
        const BlockedMemoryDesc::CmpMask outputMask =
            acceptAnyBatchStride ? BlockedMemoryDesc::EMPTY_MASK : BlockedMemoryDesc::SKIP_OFFSET_MASK;
        nodeConfig.outConfs.emplace_back(outputDesc, outputMask, -1);

        supportedPrimitiveDescriptors.emplace_back(nodeConfig, impl_desc_type::undef);
    }

    sortSupportedPrimitiveDescriptors(supportedPrimitiveDescriptors, context->getConfig().modelType);
}

bool Eltwise::created() const {
    return getType() == Type::Eltwise;
}

bool Eltwise::canFuse(const NodePtr& node) const {
    auto isIntegerComputeSupported = [](const Node* node) {
        if (none_of(node->getAlgorithm(),
                    Algorithm::EltwiseAdd,
                    Algorithm::EltwiseMultiply,
                    Algorithm::EltwiseMulAdd,
                    Algorithm::EltwiseSubtract,
                    Algorithm::EltwiseDivide,
                    Algorithm::EltwiseSquaredDifference)) {
            return false;
        }

        return all_of_values(node->getOriginalInputPrecisions(), ov::element::i32);
    };

    if (!EltwiseJitExecutor::supports(m_attrs, inputShapes.front().getRank())) {
        return false;
    }

#if defined(OPENVINO_ARCH_ARM64) || defined(OPENVINO_ARCH_RISCV64)
    const auto* eltwise = dynamic_cast<const Eltwise*>(node.get());
    if (eltwise == nullptr ||
        !EltwiseJitExecutor::supports(eltwise->attrs(), eltwise->getInputShapeAtPort(0).getRank())) {
        return false;
    }
#endif

    if (any_of(getAlgorithm(),
               Algorithm::EltwiseLog,
               Algorithm::EltwiseBitwiseAnd,
               Algorithm::EltwiseBitwiseNot,
               Algorithm::EltwiseBitwiseOr,
               Algorithm::EltwiseBitwiseXor,
               Algorithm::EltwiseBitwiseLeftShift,
               Algorithm::EltwiseBitwiseRightShift) ||
        any_of(node->getAlgorithm(),
               Algorithm::EltwiseLog,
               Algorithm::EltwiseBitwiseAnd,
               Algorithm::EltwiseBitwiseNot,
               Algorithm::EltwiseBitwiseOr,
               Algorithm::EltwiseBitwiseXor,
               Algorithm::EltwiseBitwiseLeftShift,
               Algorithm::EltwiseBitwiseRightShift)) {
        return false;
    }

    bool isIntegerNode = isIntegerComputeSupported(this);
    if (isIntegerNode && node->getType() != Type::Eltwise) {
        return false;
    }

    // FQ inputs with quantization parameters will be hided inside post_op object, so will not increase inputs number
    size_t addedInputEdgesNum = node->getType() != Type::FakeQuantize ? (node->getParentEdges().size() - 1) : 0;
    if (getParentEdges().size() + addedInputEdgesNum > MAX_ELTWISE_INPUTS) {
        return false;
    }

    if (node->getType() == Type::Eltwise) {
        // [WA] Since execution precision change from I32 to FP32 for arithmetic operations may lead to incorrect
        // results we disable fusing cases which may lead to invalid precision conversions inside the kernel [TODO] We
        // need to rewrite support for different precisions at all to avoid implicit conversions to FP32 (all should be
        // handled via explicit convert operations)
        bool isIntegerFusingNode = isIntegerComputeSupported(node.get());
        if ((isIntegerNode && !isIntegerFusingNode) || (!isIntegerNode && isIntegerFusingNode)) {
            return false;
        }

        if (node->getParentEdgeAt(0)->getParent().get() != this) {
            // Eltwise jitter doesn't respect commutative property, so fusing is disabled in case it applied not for
            // 0-th port.
            if (any_of(node->getAlgorithm(),
                       Algorithm::EltwiseSubtract,
                       Algorithm::EltwiseDivide,
                       Algorithm::EltwiseFloorMod,
                       Algorithm::EltwiseMod,
                       Algorithm::EltwisePowerDynamic,
                       Algorithm::EltwiseGreater,
                       Algorithm::EltwiseGreaterEqual,
                       Algorithm::EltwiseLess,
                       Algorithm::EltwiseLessEqual,
                       Algorithm::EltwiseMulAdd,
                       Algorithm::EltwiseSelect)) {
                return false;
            }

            // Limitation: inputs precision definition inside Eltwise node assumes fusing is applied for 0-th port,
            // otherwise we need identical precision on all inputs of fused node
            for (size_t i = 1; i < getOriginalInputsNumber(); i++) {
                if (getOriginalInputPrecisionAtPort(0) != getOriginalInputPrecisionAtPort(i)) {
                    return false;
                }
            }
        }

        // We can use optimized execution with fusions only in cases when dim rank is less or equal to the maximum
        // possible
        return node->getInputShapeAtPort(0).getRank() <= MAX_ELTWISE_DIM_RANK;
    }

    if (node->getType() == Type::FakeQuantize) {
        return node->getAlgorithm() != Algorithm::FQBinarization;
    }

    return false;
}

ov::element::Type Eltwise::getRuntimePrecision() const {
    std::vector<ov::element::Type> inputPrecisions;
    // Don't take bias precision into account
    size_t inputsNumLimit = 2;
    for (size_t i = 0; i < std::min(getParentEdges().size(), inputsNumLimit); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == Edge::Status::Validated) {
            inputPrecisions.emplace_back(
                DnnlExtensionUtils::DataTypeToElementType((parentEdge->getMemoryPtr()->getDataType())));
        }
    }

    return getMaxPrecision(inputPrecisions);
}

void Eltwise::createPrimitive() {
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        m_memory[ARG_SRC + i] = getSrcMemoryAtPort(i);
    }
    m_memory[ARG_DST] = getDstMemoryAtPort(0);

    m_executor = m_factory->make(m_memory);
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());

    Node::createPrimitive();
}

void Eltwise::prepareParams() {
    m_executor->update(m_memory);
}

void Eltwise::execute([[maybe_unused]] const dnnl::stream& strm) {
    OPENVINO_DEBUG_ASSERT(m_executor, "Eltwise executor not created");
    m_executor->execute(m_memory);
}

void Eltwise::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void Eltwise::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptor(getImplPriority(), true);
}

bool Eltwise::canBeInPlace() const {
    if (getParentEdgeAt(0)->getParent()->getType() == Type::Input) {
        return false;
    }

    for (const auto& parentEdge : getParentEdges()) {
        auto parent = parentEdge.lock()->getParent();
        if (parent->getChildEdges().size() != 1) {
            return false;
        }

        // WA to prevent memory corruption caused by inplace feature
        if (parent->getType() == Type::Concatenation) {
            for (const auto& parentParentEdge : parent->getParentEdges()) {
                auto parentParent = parentParentEdge.lock()->getParent();
                if (parentParent->getChildEdges().size() != 1) {
                    return false;
                }
            }
        }
    }

    return getInputShapeAtPort(0) == getOutputShapeAtPort(0);
}

void Eltwise::fuseInto(NodePtr& parentNode) {
    // Handle special convolution add fusing case
    m_attrs.specialConvolutionAddFusing =
        (parentNode->getType() == Type::Convolution || parentNode->getType() == Type::BinaryConvolution) &&
        getAlgorithm() == Algorithm::EltwiseAdd &&
        dimsEqualWeak(getInputShapeAtPort(0).getDims(), getInputShapeAtPort(1).getDims()) &&
        !getParentEdgeAt(0)->getParent()->isConstant() && !getParentEdgeAt(1)->getParent()->isConstant();

    if ((m_attrs.scales.empty() && m_attrs.shifts.empty()) && !m_attrs.specialConvolutionAddFusing &&
        canBePerformedAsScaleShift(parentNode.get())) {
        std::tie(m_attrs.scales, m_attrs.shifts) = getScalesAndShifts(parentNode.get());
    }

    Node::fuseInto(parentNode);
}

// Post-ops implementation methods
void Eltwise::appendMemory(const std::vector<float>& data, MemoryPtr& memPtr, std::vector<MemoryPtr>& postOpsMem) {
    if (!memPtr) {
        DnnlBlockedMemoryDesc memoryDesc(ov::element::f32, {data.size()});
        memPtr = std::make_shared<Memory>(getEngine(), memoryDesc, data.data());
        postOpsMem.push_back(memPtr);
    }
}

void Eltwise::appendMemory(const std::vector<float>& data,
                           [[maybe_unused]] MemoryPtr& memPtr,
                           std::vector<const void*>& postOpsMem) {
    postOpsMem.push_back(data.data());
}

template <typename T>
void Eltwise::appendPostOpsImpl(dnnl::post_ops& ops,
                                const VectorDims& postOpDims,
                                std::vector<T>& postOpsMem,
                                const int channelAxis) {
    if (getOneDnnAlgorithm() != dnnl::algorithm::undef) {
        switch (getOneDnnAlgorithm()) {
        case dnnl::algorithm::eltwise_relu:
        case dnnl::algorithm::eltwise_tanh:
        case dnnl::algorithm::eltwise_elu:
        case dnnl::algorithm::eltwise_square:
        case dnnl::algorithm::eltwise_abs:
        case dnnl::algorithm::eltwise_sqrt:
        case dnnl::algorithm::eltwise_linear:
        case dnnl::algorithm::eltwise_soft_relu:
        case dnnl::algorithm::eltwise_logistic:
        case dnnl::algorithm::eltwise_exp:
        case dnnl::algorithm::eltwise_gelu_erf:
        case dnnl::algorithm::eltwise_gelu_tanh:
        case dnnl::algorithm::eltwise_clip:
        case dnnl::algorithm::eltwise_swish:
        case dnnl::algorithm::eltwise_hardswish:
        case dnnl::algorithm::eltwise_mish:
        case dnnl::algorithm::eltwise_hsigmoid:
        case dnnl::algorithm::eltwise_round_half_to_even:
        case dnnl::algorithm::eltwise_round_half_away_from_zero:
            ops.append_eltwise(getOneDnnAlgorithm(), getAlpha(), getBeta());
            break;
        default:
            CPU_NODE_THROW("Appending Eltwise node with name '", getName(), "' as post operation is not supported");
        }

        return;
    }
    // per-tensor EltwisePowerStatic can be implemented with more well-supported eltwise postOps
    if (getAlgorithm() == Algorithm::EltwisePowerStatic) {
        // d = s*beta + gamma
        ops.append_eltwise(dnnl::algorithm::eltwise_linear, getBeta(), getGamma());
        if (getAlpha() != 1.0F) {
            // d = 1 * s^alpha
            ops.append_eltwise(dnnl::algorithm::eltwise_pow, 1.0F, getAlpha());
        }
        return;
    }
    size_t channelSize = 1;
    if (channelAxis >= 0) {
        const auto chIdx = postOpDims.size() > 1 ? channelAxis : 0;
        channelSize = postOpDims[chIdx];
    }
    // since legacy depthwise post ops mechanism requires broadcasted data we need to reinitilize it in case of
    // changed shape
    if (m_depthwiseData.empty() || m_depthwiseDataSize != 2 * channelSize) {
        m_depthwiseData.clear();
        m_depthwiseMemory.reset();

        m_depthwiseData.insert(m_depthwiseData.end(), getScales().begin(), getScales().end());
        if (getScales().size() == 1) {
            m_depthwiseData.resize(channelSize, m_depthwiseData.back());
        } else if (getScales().size() != channelSize) {
            CPU_NODE_THROW("Appending node has failed due to scales data size inconsistency");
        }
        m_depthwiseData.insert(m_depthwiseData.end(), getShifts().begin(), getShifts().end());
        if (getShifts().empty()) {
            // in case of Prelu algorithm scales data is always empty
            m_depthwiseData.resize(2 * channelSize, 0);
        } else if (getShifts().size() == 1) {
            m_depthwiseData.resize(2 * channelSize, m_depthwiseData.back());
        } else if (getShifts().size() != channelSize) {
            CPU_NODE_THROW("Appending node has failed due to shifts data size inconsistency");
        }
        m_depthwiseDataSize = 2 * channelSize;

        // always align for legacy scale/shift post ops
        constexpr int bufferAlignment = 16;
        int bufferPaddingSize = rnd_up(channelSize, bufferAlignment) - channelSize;
        m_depthwiseData.resize(m_depthwiseDataSize + bufferPaddingSize, 0);
    }

    CPU_NODE_ASSERT(!m_depthwiseData.empty(), "cannot be performed since buffers are not allocated");

    std::array<size_t, 2> offsets = {0};
    offsets[1] = offsets[0] + channelSize;

    /* @todo legacy depthwise post ops are kept for now
     * for performance reasons
     */
    switch (getAlgorithm()) {
    case Algorithm::EltwiseAdd:
    case Algorithm::EltwiseSubtract:
    case Algorithm::EltwiseMultiply:
    case Algorithm::EltwiseDivide:
    case Algorithm::EltwiseMulAdd:
    case Algorithm::EltwisePowerStatic:
        ops.append_depthwise(dnnl::algorithm::depthwise_scale_shift, offsets);
        break;
    case Algorithm::EltwisePrelu:
        ops.append_depthwise(dnnl::algorithm::depthwise_prelu, offsets);
        break;
    default:
        CPU_NODE_THROW("as post operation is not supported");
    }

    appendMemory(m_depthwiseData, m_depthwiseMemory, postOpsMem);
}

void Eltwise::appendPostOps(dnnl::post_ops& ops,
                            const VectorDims& postOpDims,
                            std::unordered_map<int, MemoryPtr>& postOpsMem,
                            const int channelAxis) {
    std::vector<MemoryPtr> postOpsMemPtrs;
    appendPostOpsImpl(ops, postOpDims, postOpsMemPtrs, channelAxis);

    CPU_NODE_ASSERT(postOpsMemPtrs.size() <= 1, "at most 1 post ops memory args can be appended.");

    if (!postOpsMemPtrs.empty()) {
        postOpsMem[DNNL_ARG_ATTR_MULTIPLE_POST_OP(ops.len() - 1) | DNNL_ARG_SRC_1] = postOpsMemPtrs[0];
    }
}

void Eltwise::appendPostOps(dnnl::post_ops& ops,
                            const VectorDims& postOpDims,
                            std::vector<const void*>& postOpsMem,
                            const int channelAxis) {
    appendPostOpsImpl(ops, postOpDims, postOpsMem, channelAxis);
}

bool Eltwise::appendAttrPostOps(DnnlPostOpsComposerLegacy& dnnlpoc,
                                bool isLastPostOp,
                                [[maybe_unused]] dnnl::memory::data_type outDataType,
                                bool allowBinary) {
    if (getOneDnnAlgorithm() != dnnl::algorithm::undef) {
        switch (getOneDnnAlgorithm()) {
        case dnnl::algorithm::eltwise_relu:
        case dnnl::algorithm::eltwise_tanh:
        case dnnl::algorithm::eltwise_elu:
        case dnnl::algorithm::eltwise_square:
        case dnnl::algorithm::eltwise_abs:
        case dnnl::algorithm::eltwise_sqrt:
        case dnnl::algorithm::eltwise_soft_relu:
        case dnnl::algorithm::eltwise_logistic:
        case dnnl::algorithm::eltwise_exp:
        case dnnl::algorithm::eltwise_gelu_erf:
        case dnnl::algorithm::eltwise_gelu_tanh:
        case dnnl::algorithm::eltwise_clip:
        case dnnl::algorithm::eltwise_swish:
        case dnnl::algorithm::eltwise_hardswish:
        case dnnl::algorithm::eltwise_mish:
        case dnnl::algorithm::eltwise_hsigmoid:
        case dnnl::algorithm::eltwise_round_half_to_even:
        case dnnl::algorithm::eltwise_round_half_away_from_zero:
            dnnlpoc.appendEltwise(getOneDnnAlgorithm(), getAlpha(), getBeta());
            break;
        case dnnl::algorithm::eltwise_linear:
            // call dnnlpoc's specialized API to generate optimized postOps sequence
            dnnlpoc.appendLinear({getAlpha()}, {getBeta()}, isLastPostOp);
            break;
        default:
            CPU_NODE_THROW("as post operation is not supported");
        }
    } else {
        switch (getAlgorithm()) {
        case Algorithm::EltwiseAdd:
        case Algorithm::EltwiseSubtract:
            return dnnlpoc.appendShift(m_attrs.shifts, allowBinary);
        case Algorithm::EltwiseDivide:
        case Algorithm::EltwiseMultiply:
            return dnnlpoc.appendScale(m_attrs.scales, isLastPostOp, allowBinary);
        case Algorithm::EltwiseMulAdd:
            return dnnlpoc.appendLinear(m_attrs.scales, m_attrs.shifts, isLastPostOp, allowBinary);
        case Algorithm::EltwisePowerStatic:
            if (m_attrs.data.beta != 1.0F && m_attrs.data.gamma != 0.0F) {
                return dnnlpoc.appendLinear(m_attrs.scales, m_attrs.shifts, isLastPostOp, allowBinary);
            } else if (m_attrs.data.beta != 1.0F) {  // Multiply if has m_attrs.scales
                return dnnlpoc.appendScale(m_attrs.scales, isLastPostOp, allowBinary);
            } else if (m_attrs.data.gamma != 0.0F) {  // Add only if has m_attrs.shifts
                return dnnlpoc.appendShift(m_attrs.shifts, allowBinary);
            }
            break;
        case Algorithm::EltwisePrelu:
            if (!allowBinary) {
                return false;
            }
            dnnlpoc.appendBinary(dnnl::algorithm::binary_prelu, m_attrs.scales);
            break;
        default:
            CPU_NODE_THROW("as post operation is not supported");
        }
    }
    return true;
}

bool Eltwise::canFuseParent(const NodePtr& parentNode) const {
#if defined(OPENVINO_ARCH_ARM64)
    if (parentNode->getType() != Type::Convert) {
        return false;
    }
    const auto& input_precisions = parentNode->getOriginalInputPrecisions();

    return EltwiseJitExecutor::supports(m_attrs, inputShapes.front().getRank(), input_precisions);
#else
    const auto isSuitableParentNode = [](const Node* parentNode) {
        return parentNode->getType() == Type::Convert &&
               (parentNode->getOriginalInputPrecisionAtPort(0) == ov::element::u8 ||
                parentNode->getOriginalInputPrecisionAtPort(0) == ov::element::i8) &&
               parentNode->getOriginalOutputPrecisionAtPort(0) == ov::element::f32;
    };

    auto isSuitableChildNode = [](const Node* childNode) {
        return childNode->getParentEdges().size() != 2;
    };

    return isSuitableParentNode(parentNode.get()) && isSuitableChildNode(this);
#endif
}

bool Eltwise::canFuseConvert(const NodePtr& convertNode) {
    if (none_of(convertNode->getOriginalOutputPrecisionAtPort(0),
                ov::element::i8,
                ov::element::u8,
                ov::element::f16,
                ov::element::bf16,
                ov::element::f32)) {
        return false;
    }
    // Convert can be fused into Eltwise only if jit implementation is supported
    return EltwiseJitExecutor::supports(m_attrs,
                                        inputShapes.front().getRank(),
                                        {},
                                        {convertNode->getOriginalOutputPrecisionAtPort(0)});
}

}  // namespace ov::intel_cpu::node
