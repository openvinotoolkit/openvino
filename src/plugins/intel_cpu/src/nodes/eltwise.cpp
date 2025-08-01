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

namespace {

struct EltwiseKey {
    std::vector<EltwiseData> eltwise_data;
    std::vector<Type> ops_list;
    VectorDims outBlkDims;
    VectorDims outOrder;
    std::vector<VectorDims> inpDims;
    std::vector<ov::element::Type> inpPrc;
    ov::element::Type outPrc;
    dnnl::post_ops postOps;
    EltwiseImplType implType;

    [[nodiscard]] size_t hash() const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;
        size_t seed = 0;
        auto hash_combine_eltwiseData = [](size_t seed, const EltwiseData& eltwiseData) {
            seed = hash_combine(seed, eltwiseData.algo);
            seed = hash_combine(seed, eltwiseData.onednnAlgorithm);
            seed = hash_combine(seed, eltwiseData.alpha);
            seed = hash_combine(seed, eltwiseData.beta);
            seed = hash_combine(seed, eltwiseData.gamma);
            return seed;
        };
        std::for_each(eltwise_data.begin(), eltwise_data.end(), [&](const EltwiseData& item) {
            seed = hash_combine_eltwiseData(seed, item);
        });
        seed = get_vector_hash(seed, ops_list);
        if (implType == EltwiseImplType::optimizedShapeAgnostic) {
            seed = hash_combine(seed, outBlkDims.back() == 1);
            for (auto&& item : inpDims) {
                seed = hash_combine(seed, item.back() == 1);
            }
        } else {
            seed = get_vector_hash(seed, outOrder);
            seed = get_vector_hash(seed, outBlkDims);
            for (auto&& item : inpDims) {
                seed = get_vector_hash(seed, item);
            }
        }
        std::for_each(inpPrc.begin(), inpPrc.end(), [&](const ov::element::Type& item) {
            seed = hash_combine(seed, item.hash());
        });
        seed = hash_combine(seed, outPrc.hash());
        seed = get_post_op_hash(seed, *postOps.get());
        seed = hash_combine(seed, implType);
        return seed;
    }

    bool operator==(const EltwiseKey& rhs) const {
        if (inpDims.size() != rhs.inpDims.size()) {
            return false;
        }

        bool result = eltwise_data == rhs.eltwise_data && ops_list == rhs.ops_list && inpPrc == rhs.inpPrc &&
                      outPrc == rhs.outPrc && *postOps.get() == *rhs.postOps.get() && implType == rhs.implType;

        if (result) {
            if (implType == EltwiseImplType::optimizedShapeAgnostic) {
                bool broadcast = false;
                bool rhsBroadcast = false;
                for (size_t i = 0; i < inpDims.size(); ++i) {
                    broadcast = (inpDims[i].back() == 1);
                    rhsBroadcast = (rhs.inpDims[i].back() == 1);
                    if (broadcast != rhsBroadcast) {
                        return false;
                    }
                }
            } else {
                result = result && outOrder == rhs.outOrder && outBlkDims == rhs.outBlkDims;
                for (size_t i = 0; i < inpDims.size() && result; ++i) {
                    result = result && (inpDims[i] == rhs.inpDims[i]);
                }
            }
        }

        return result;
    }
};

class EltwiseJitExecutor : public Eltwise::IEltwiseExecutor {
public:
    static void offset_out_calc(VectorDims& offset, const VectorDims& dims) {
        int k = 1;
        for (int i = offset.size() - 1; i >= 0; i--) {
            offset[i] = k;
            k *= dims[i];
        }
    }

    static void offset_in_calc(VectorDims& offset, const VectorDims& dims_in, const VectorDims& dims_out) {
        int k = 1;
        for (int i = offset.size() - 1; i >= 0; i--) {
            offset[i] = (dims_in[i] == dims_out[i]) ? k : 0;
            k *= dims_in[i];
        }
    }

    EltwiseJitExecutor(const std::vector<EltwiseData>& eltwise_data,
                       const std::vector<Type>& ops_list,
                       const VectorDims& outBlkDims,
                       const VectorDims& outOrder,
                       std::vector<VectorDims> inpDims,
                       const std::vector<ov::element::Type>& inpPrc,
                       const ov::element::Type& outPrc,
                       const dnnl::post_ops& post_ops,
                       bool useRuntimePtrs) {
        auto collapseLastDims = [](std::vector<size_t>& dims, int dimsToCollapse) {
            for (size_t i = dims.size() - 2; i > dims.size() - dimsToCollapse - 2; i--) {
                dims[dims.size() - 1] *= dims[i];
            }

            for (int i = dims.size() - 2; i >= dimsToCollapse; i--) {
                dims[i] = dims[i - dimsToCollapse];
            }

            for (int i = dimsToCollapse - 1; i >= 0; i--) {
                dims[i] = 1;
            }
        };

        auto collapseLastOffsets = [](std::vector<size_t>& dims, int dimsToCollapse) {
            for (size_t i = dims.size() - 2; i > dims.size() - dimsToCollapse - 2; i--) {
                if (dims[dims.size() - 1] > 0 || dims[i] > 0) {
                    dims[dims.size() - 1] = std::max(dims[dims.size() - 1], static_cast<size_t>(1)) *
                                            std::max(dims[i], static_cast<size_t>(1));
                } else {
                    dims[dims.size() - 1] *= dims[i];
                }
            }

            for (int i = dims.size() - 2; i >= dimsToCollapse; i--) {
                dims[i] = dims[i - dimsToCollapse];
            }

            for (int i = dimsToCollapse - 1; i >= 0; i--) {
                dims[i] = 0;
            }
        };

        auto isFusedWith = [&](Type type_) {
            auto start_itr = ops_list.begin();
            std::advance(start_itr, 1);  // apply offset since the first op in the list is the op itself
            return any_of(start_itr, ops_list.end(), [=](Type type) {
                return type == type_;
            });
        };

        if (inpDims.empty()) {
            OPENVINO_THROW("Can not make Eltwise executor from empty input dims array");
        } else if (inpDims.front().empty()) {
            OPENVINO_THROW("Can not make Eltwise executor from empty input dims members");
        }

        jit_eltwise_params jep = {};
        size_t inputsNumber = inpDims.size();

        jep.use_runtime_ptrs = useRuntimePtrs;

        jep.input_size = inpDims.front().size();

        jep.dims.resize(jep.input_size, 1);

        if (outBlkDims.empty()) {
            OPENVINO_THROW("Can not make Eltwise executor from empty block dims vector");
        }

        size_t outRank = outBlkDims.size();
        for (size_t i = 0; i < outRank; i++) {
            jep.dims[jep.dims.size() - 1 - i] = outBlkDims[outRank - 1 - i];
        }

        for (auto& inpDim : inpDims) {
            for (size_t j = 0; j < inpDim.size(); j++) {
                if (inpDim[j] != jep.dims[j] && inpDim[j] != 1) {
                    OPENVINO_THROW("Eltwise executor got invalid input/output dims configuration.");
                }
            }
        }

        if (outBlkDims.size() != outOrder.size()) {
            OPENVINO_THROW(
                "Can not make Eltwise executor due to out blocked dims and out order vectors size mismatch.");
        }

        int lastUnchangedAxis = 0;
        size_t oc_size = 0;
        jep.oc_offsets.resize(jep.input_size, 0);
        std::fill(jep.oc_offsets.begin(), jep.oc_offsets.end(), 0);
        if (isFusedWith(Type::FakeQuantize)) {
            size_t offset_oc = 1;
            for (int i = outOrder.size() - 1; i >= 0; i--) {
                if (outOrder[i] == 1) {
                    int oc_dim_idx = i + (jep.input_size - outOrder.size());
                    jep.oc_offsets[oc_dim_idx] = offset_oc;
                    offset_oc *= jep.dims[oc_dim_idx];
                    if (oc_dim_idx + 1 !=
                        static_cast<int>(jep.input_size)) {  // since in nspc case we can safely collapse the last axis
                        lastUnchangedAxis = oc_dim_idx;
                    }
                }
            }
            oc_size = jep.oc_offsets[jep.dims.size() - 1] != 0 ? jep.dims[jep.dims.size() - 1] : 1;
        }

        int maxCollapsedDims = static_cast<int>(jep.dims.size()) - lastUnchangedAxis - 2;

        size_t fullWorkAmount = 1;
        for (size_t dim : jep.dims) {
            fullWorkAmount *= dim;
        }

        m_threads_num = static_cast<size_t>(parallel_get_max_threads());
        size_t minimalJitWorkAmount = 256;
        size_t currentJitWorkAmount = jep.dims[jep.dims.size() - 1];
        int collapsedDims = 0;

        bool hasDifferentDims = false;
        while (!useRuntimePtrs && currentJitWorkAmount < minimalJitWorkAmount &&
               currentJitWorkAmount < fullWorkAmount) {
            if (collapsedDims >= maxCollapsedDims) {
                break;
            }

            for (size_t j = 1; j < inpDims.size(); j++) {
                if (inpDims[j].back() != inpDims[0].back()) {
                    hasDifferentDims = true;
                    break;
                }
            }

            if (oc_size > 1 && oc_size != inpDims[0][inpDims[0].size() - 1]) {
                hasDifferentDims = true;
            }

            bool canCollapse = true;
            for (auto& inpDim : inpDims) {
                if (inpDim[inpDim.size() - 2] != 1) {
                    if (hasDifferentDims) {
                        canCollapse = false;
                        break;
                    }
                }
            }

            if (!canCollapse) {
                break;
            }

            size_t nextJitWorkAmount = currentJitWorkAmount * jep.dims[jep.dims.size() - 2];
            if (fullWorkAmount / nextJitWorkAmount >= m_threads_num) {
                currentJitWorkAmount = nextJitWorkAmount;
                collapsedDims++;

                for (auto& inpDim : inpDims) {
                    collapseLastDims(inpDim, 1);
                }
                collapseLastDims(jep.dims, 1);

                if (isFusedWith(Type::FakeQuantize)) {
                    collapseLastOffsets(jep.oc_offsets, 1);
                }
            } else {
                break;
            }
        }

        if (inpPrc.size() != inputsNumber) {
            OPENVINO_THROW("Can not make Eltwise executor. Wrong input precisions vector size.");
        }

        if (!useRuntimePtrs) {
            _batchDimIdx = jep.input_size - outBlkDims.size() + collapsedDims;
            _schedulerWorkAmount = fullWorkAmount / jep.dims[jep.dims.size() - 1];

            // init offset
            jep.dst_offsets.resize(jep.input_size, 1);
            offset_out_calc(jep.dst_offsets, jep.dims);
            for (size_t j = 0; j < jep.input_size; j++) {
                jep.dst_offsets[j] *= outPrc.size();
            }

            for (size_t i = 0; i < inputsNumber; i++) {
                jep.src_offsets[i].resize(jep.input_size, 1);
                offset_in_calc(jep.src_offsets[i], inpDims[i], jep.dims);
                for (size_t j = 0; j < jep.input_size; j++) {
                    jep.src_offsets[i][j] *= inpPrc[i].size();
                }
            }
        }

        jep.inputs_number = inputsNumber;

        for (size_t i = 0; i < inputsNumber; i++) {
            jep.src_prc[i] = inpPrc[i];
            jep.src_size[i] = inpDims[i][inpDims[i].size() - 1];
        }
        jep.dst_prc = outPrc;
        jep.work_amount = jep.dst_size = jep.dims.back();
        jep.oc_size = oc_size;

        std::transform(jep.oc_offsets.begin(), jep.oc_offsets.end(), jep.oc_offsets.begin(), [](size_t& offset) {
            return offset * sizeof(float);
        });

#if defined(OPENVINO_ARCH_X86_64)
        if (mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
            _pKernel = std::make_unique<jit_uni_eltwise_generic<dnnl::impl::cpu::x64::avx512_core>>(jep,
                                                                                                    eltwise_data,
                                                                                                    ops_list,
                                                                                                    post_ops);
        } else if (mayiuse(dnnl::impl::cpu::x64::avx2)) {
            _pKernel = std::make_unique<jit_uni_eltwise_generic<dnnl::impl::cpu::x64::avx2>>(jep,
                                                                                             eltwise_data,
                                                                                             ops_list,
                                                                                             post_ops);
        } else if (mayiuse(dnnl::impl::cpu::x64::sse41)) {
            _pKernel = std::make_unique<jit_uni_eltwise_generic<dnnl::impl::cpu::x64::sse41>>(jep,
                                                                                              eltwise_data,
                                                                                              ops_list,
                                                                                              post_ops);
        } else {
            OPENVINO_THROW("Can't create jit eltwise kernel");
        }
#elif defined(OPENVINO_ARCH_ARM64)
        if (mayiuse(aarch64::asimd)) {
            _pKernel = std::make_unique<jit_uni_eltwise_generic<aarch64::asimd>>(jep, eltwise_data, ops_list, post_ops);
        } else {
            OPENVINO_THROW("Can't create jit eltwise kernel");
        }
#elif defined(OPENVINO_ARCH_RISCV64)
        (void)post_ops;
        if (mayiuse(ov::intel_cpu::riscv64::gv)) {
            _pKernel = std::make_unique<ov::intel_cpu::riscv64::jit_uni_eltwise_generic<ov::intel_cpu::riscv64::gv>>(
                jep,
                eltwise_data);
        } else {
            OPENVINO_THROW("Can't create jit eltwise kernel");
        }
#endif  // OPENVINO_ARCH_RISCV64
        if (_pKernel) {
            _pKernel->create_ker();
        }
    }

    void exec(const jit_eltwise_call_args_ptrs& args_ptrs, const VectorDims& dims_out) override {
        if (!_pKernel) {
            OPENVINO_THROW("Can't execute, kernel for eltwise node is not compiled");
        }

        if (_pKernel->jep_.input_size == optimalTensorRank) {
            // execute Optimized 6D
            auto d6_loop = [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                auto args = jit_eltwise_call_args_indexes();
                args.indexes[0] = i0;
                args.indexes[1] = i1;
                args.indexes[2] = i2;
                args.indexes[3] = i3;
                args.indexes[4] = i4;

                (*_pKernel)(&args_ptrs, &args);
            };

            parallel_nt_static(m_threads_num, [&](const int ithr, const int nthr) {
                for_5d(ithr, nthr, dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], d6_loop);
            });
        } else {
            // execute Optimized Generic
            if (_pKernel->jep_.use_runtime_ptrs) {
                // recalculate _schedulerWorkAmount
                _schedulerWorkAmount = 1;
                for (size_t i = 0; i < dims_out.size() - 1; i++) {
                    _schedulerWorkAmount *= dims_out[i];
                }
            }
            parallel_nt(m_threads_num, [&](const int ithr, const int nthr) {
                size_t start = 0;
                size_t end = 0;
                splitter(_schedulerWorkAmount, nthr, ithr, start, end);

                std::vector<size_t> counters(dims_out.size() - 1, 0);
                auto args = jit_eltwise_call_args_indexes();
                for (size_t iwork = start; iwork < end; ++iwork) {
                    size_t tmp = iwork;
                    for (ptrdiff_t j = dims_out.size() - 2; j >= 0; j--) {
                        counters[j] = tmp % dims_out[j];
                        tmp /= dims_out[j];
                    }

                    for (size_t j = 0; j < counters.size(); j++) {
                        args.indexes[j] = counters[j];
                    }

                    (*_pKernel)(&args_ptrs, &args);
                }
            });
        }
    }
    [[nodiscard]] const VectorDims& getOutDims() const override {
        if (!_pKernel) {
            OPENVINO_THROW("Can't get jit eltwise params, kernel for Eltwise executor is not compiled");
        }
        return _pKernel->jep_.dims;
    }
    [[nodiscard]] size_t getBatchDimIdx() const override {
        return _batchDimIdx;
    }

    static bool isSupportedOp(const Node* node,
                              [[maybe_unused]] const float alpha,
                              [[maybe_unused]] const float beta,
                              [[maybe_unused]] const float gamma,
                              [[maybe_unused]] const ov::element::TypeVector& input_precisions = {},
                              [[maybe_unused]] const ov::element::TypeVector& output_precisions = {}) {
#if defined(OPENVINO_ARCH_X86_64)
        const auto isISASupportedByJIT = mayiuse(dnnl::impl::cpu::x64::sse41);
#elif defined(OPENVINO_ARCH_ARM64)
        const auto isISASupportedByJIT = mayiuse(dnnl::impl::cpu::aarch64::asimd);
#elif defined(OPENVINO_ARCH_RISCV64)
        const auto isISASupportedByJIT = mayiuse(ov::intel_cpu::riscv64::gv);
#else
        const auto isISASupportedByJIT = false;
#endif
        // if dim rank is greater than the maximum possible, we should not use JIT execution
        if (!isISASupportedByJIT || node->getInputShapeAtPort(0).getRank() > MAX_ELTWISE_DIM_RANK) {
            return false;
        }

        const auto algorithm = node->getAlgorithm();
        if (one_of(algorithm,
                   Algorithm::EltwiseLog,
                   Algorithm::EltwiseBitwiseLeftShift,
                   Algorithm::EltwiseBitwiseRightShift)) {
            return false;  // NOLINT(readability-simplify-boolean-expr) since no further checks on x64 are required
        }

#if defined(OPENVINO_ARCH_X86_64)
        return true;

#elif defined(OPENVINO_ARCH_ARM64)
        if (one_of(algorithm,
                   Algorithm::EltwiseBitwiseAnd,
                   Algorithm::EltwiseBitwiseNot,
                   Algorithm::EltwiseBitwiseOr,
                   Algorithm::EltwiseBitwiseXor)) {
            return false;
        }

        std::set<ov::element::Type> supported_input_precisions = std::set<ov::element::Type>{ov::element::f16,
                                                                                             ov::element::f32,
                                                                                             ov::element::i32,
                                                                                             ov::element::i8,
                                                                                             ov::element::u8};

        std::set<ov::element::Type> supported_output_precisions = supported_input_precisions;
        if (one_of(algorithm, Algorithm::EltwiseDivide, Algorithm::EltwiseFloor)) {
            supported_input_precisions = std::set<ov::element::Type>{ov::element::f16, ov::element::f32};
        }

        auto fusedOps = node->getFusedWith();
        if (!fusedOps.empty()) {
            // Divide and Floor (issue #138629) operations are supported for fp32 and fp16 only.
            if (one_of(fusedOps.back()->getAlgorithm(), Algorithm::EltwiseDivide, Algorithm::EltwiseFloor)) {
                supported_output_precisions = std::set<ov::element::Type>{ov::element::f16, ov::element::f32};
            }
        } else {
            supported_output_precisions = supported_input_precisions;
        }

#elif defined(OPENVINO_ARCH_RISCV64)
        if (!one_of(algorithm,
                    Algorithm::EltwiseAbs,
                    Algorithm::EltwiseAdd,
                    Algorithm::EltwiseClamp,
                    Algorithm::EltwiseDivide,
                    Algorithm::EltwiseExp,
                    Algorithm::EltwiseFloor,
                    Algorithm::EltwiseGreater,
                    Algorithm::EltwiseGreaterEqual,
                    Algorithm::EltwiseLessEqual,
                    Algorithm::EltwiseLogicalAnd,
                    Algorithm::EltwiseLogicalNot,
                    Algorithm::EltwiseLogicalXor,
                    Algorithm::EltwiseMaximum,
                    Algorithm::EltwiseMinimum,
                    Algorithm::EltwiseMod,
                    Algorithm::EltwiseMulAdd,
                    Algorithm::EltwiseMultiply,
                    Algorithm::EltwiseNegative,
                    Algorithm::EltwiseNotEqual,
                    Algorithm::EltwisePowerStatic,
                    Algorithm::EltwisePrelu,
                    Algorithm::EltwiseRelu,
                    Algorithm::EltwiseSigmoid,
                    Algorithm::EltwiseSqrt,
                    Algorithm::EltwiseSubtract)) {
            return false;
        }

        const std::set<ov::element::Type> supported_input_precisions = {ov::element::f32,
                                                                        ov::element::i32,
                                                                        ov::element::i8,
                                                                        ov::element::u8};
        auto supported_output_precisions = supported_input_precisions;
#endif

#if defined(OPENVINO_ARCH_ARM64) || defined(OPENVINO_ARCH_RISCV64)
        const auto check_precisions = [&](const std::vector<ov::element::Type>& input_precisions,
                                          const std::vector<ov::element::Type>& output_precisions) {
            if (std::any_of(input_precisions.begin(),
                            input_precisions.end(),
                            [&supported_input_precisions](const ov::element::Type& precision) {
                                return supported_input_precisions.find(precision) == supported_input_precisions.end();
                            })) {
                return false;
            }

            return !std::any_of(output_precisions.begin(),
                                output_precisions.end(),
                                [&supported_output_precisions](const ov::element::Type& precision) {
                                    return supported_output_precisions.find(precision) ==
                                           supported_output_precisions.end();
                                });
        };

        auto out_precisions = output_precisions.empty() ? node->getOriginalOutputPrecisions() : output_precisions;
        return check_precisions(input_precisions, out_precisions);
#endif

        // Unsupported architectures should return false:
        return false;
    }

private:
    std::unique_ptr<jit_uni_eltwise_kernel> _pKernel;
    size_t _schedulerWorkAmount = 0;
    size_t _batchDimIdx = 0;
    size_t m_threads_num = 0LU;

public:
    static const int optimalTensorRank = 6;
};

/* enabled only for float at float16_t at the moment
 * can be extended in the future */
template <typename T>
class EltwiseRefBaseExecutor : public Eltwise::IEltwiseExecutor {
public:
    EltwiseRefBaseExecutor(const EltwiseData& opData,
                           const VectorDims& outBlkDims,
                           const std::vector<VectorDims>& inpDims)
        : _opData(opData),
          _inpDims(inpDims) {
        if (inpDims.empty()) {
            OPENVINO_THROW("Can not make Eltwise executor from empty input dims array");
        } else if (inpDims.front().empty()) {
            OPENVINO_THROW("Can not make Eltwise executor from empty input dims array members");
        }

        if (outBlkDims.empty()) {
            OPENVINO_THROW("Can not make Eltwise executor from empty output blocked dims vector");
        }

        _inputNum = inpDims.size();
        size_t input_size = inpDims.front().size();
        _batchDimIdx = input_size - outBlkDims.size();

        _dims.resize(input_size, 1);
        for (size_t i = 0; i < outBlkDims.size(); i++) {
            _dims[_dims.size() - 1 - i] = outBlkDims[outBlkDims.size() - 1 - i];
        }

        _fullWorkAmount = 1;
        for (size_t _dim : _dims) {
            _fullWorkAmount *= _dim;
        }

        // init offset
        _dst_offsets.resize(input_size, 1);
        EltwiseJitExecutor::offset_out_calc(_dst_offsets, _dims);
        for (size_t j = 0; j < input_size; j++) {
            _dst_offsets[j] *= sizeof(T);
        }

        for (size_t i = 0; i < _inputNum; i++) {
            _src_offsets[i].resize(input_size, 1);
            EltwiseJitExecutor::offset_in_calc(_src_offsets[i], inpDims[i], _dims);
            for (size_t j = 0; j < input_size; j++) {
                _src_offsets[i][j] *= sizeof(T);
            }
        }
    }

    [[nodiscard]] const VectorDims& getOutDims() const override {
        return _dims;
    }

    [[nodiscard]] size_t getBatchDimIdx() const override {
        return _batchDimIdx;
    }

protected:
    void init_ptr(const jit_eltwise_call_args_ptrs& args_ptrs,
                  const VectorDims& dims_out,
                  std::vector<size_t>& counters,
                  const size_t iwork,
                  std::vector<T>& src_f,
                  T*& dst_ptr_f) {
        size_t tmp = iwork;
        for (ptrdiff_t j = dims_out.size() - 1; j >= 0; j--) {
            counters[j] = tmp % dims_out[j];
            tmp /= dims_out[j];
        }

        size_t index_in[MAX_ELTWISE_INPUTS] = {0};
        for (size_t i = 0; i < _inputNum; i++) {
            index_in[i] = 0;
            for (size_t j = 0; j < counters.size(); j++) {
                index_in[i] += counters[j] * _src_offsets[i][j];
            }
            index_in[i] /= sizeof(T);
        }

        size_t index_out = 0;
        for (size_t j = 0; j < counters.size(); j++) {
            index_out += counters[j] * _dst_offsets[j];
        }
        index_out /= sizeof(T);

        // std::vector<T> src_f(_inputNum);
        for (size_t i = 0; i < _inputNum; i++) {
            src_f[i] = (reinterpret_cast<const T*>(args_ptrs.src_ptr[i]) + index_in[i])[0];
        }
        dst_ptr_f = reinterpret_cast<T*>(args_ptrs.dst_ptr) + index_out;
    }

    const EltwiseData _opData;
    VectorDims _dims;
    VectorDims _src_offsets[MAX_ELTWISE_INPUTS];
    VectorDims _dst_offsets;
    size_t _fullWorkAmount = 0;
    size_t _inputNum = 0;
    size_t _batchDimIdx = 0;
    std::vector<VectorDims> _inpDims;
};

/* enabled only for float at float16_t at the moment
 * can be extended in the future */
template <typename T, std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, dnnl::impl::float16_t>>* = nullptr>
class EltwiseRefExecutor : public EltwiseRefBaseExecutor<T> {
public:
    EltwiseRefExecutor(const EltwiseData& opData, const VectorDims& outBlkDims, std::vector<VectorDims> inpDims)
        : EltwiseRefBaseExecutor<T>(opData, outBlkDims, inpDims) {}

    void exec(const jit_eltwise_call_args_ptrs& args_ptrs, const VectorDims& dims_out) override {
        if (this->_opData.algo == Algorithm::EltwiseLog) {
            const T* src_ptr_f = reinterpret_cast<const T*>(args_ptrs.src_ptr[0]);
            T* dst_ptr_f = reinterpret_cast<T*>(args_ptrs.dst_ptr);
            parallel_for(this->_fullWorkAmount, [&](size_t i) {
                dst_ptr_f[i] = logf(src_ptr_f[i]);
            });
            return;
        }
        if (this->_opData.algo == Algorithm::EltwisePowerStatic) {
            const T* src_ptr_f = reinterpret_cast<const T*>(args_ptrs.src_ptr[0]);
            T* dst_ptr_f = reinterpret_cast<T*>(args_ptrs.dst_ptr);
            if (this->_opData.alpha == 2) {
                parallel_for(this->_fullWorkAmount, [&](size_t i) {
                    dst_ptr_f[i] = (this->_opData.beta * src_ptr_f[i] + this->_opData.gamma) *
                                   (this->_opData.beta * src_ptr_f[i] + this->_opData.gamma);
                });
            } else {
                parallel_for(this->_fullWorkAmount, [&](size_t i) {
                    dst_ptr_f[i] = powf(this->_opData.beta * src_ptr_f[i] + this->_opData.gamma, this->_opData.alpha);
                });
            }
            return;
        }
        if (this->_opData.algo == Algorithm::EltwisePowerDynamic) {
            const T* src_ptr_f = reinterpret_cast<const T*>(args_ptrs.src_ptr[0]);
            const T* src_ptr_f_pow = reinterpret_cast<const T*>(args_ptrs.src_ptr[1]);
            T* dst_ptr_f = reinterpret_cast<T*>(args_ptrs.dst_ptr);

            uint32_t count_of_power_values = 1;
            for (uint64_t i : this->_inpDims[1]) {
                count_of_power_values *= i;
            }

            if (count_of_power_values == 1) {
                if (src_ptr_f_pow[0] != 2) {
                    parallel_for(this->_fullWorkAmount, [&](size_t i) {
                        dst_ptr_f[i] = powf(src_ptr_f[i], src_ptr_f_pow[0]);
                    });
                } else {
                    parallel_for(this->_fullWorkAmount, [&](size_t i) {
                        dst_ptr_f[i] = src_ptr_f[i] * src_ptr_f[i];
                    });
                }
                return;
            }
        }

        std::shared_ptr<ref_eltwise_scalar_fwd_t> ref_eltwise_injector = nullptr;
        if (this->_opData.onednnAlgorithm != dnnl::algorithm::undef) {
            ref_eltwise_injector =
                std::make_shared<ref_eltwise_scalar_fwd_t>(static_cast<dnnl_alg_kind_t>(this->_opData.onednnAlgorithm),
                                                           this->_opData.alpha,
                                                           this->_opData.beta,
                                                           1.F);
        }

        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0;
            size_t end = 0;
            splitter(this->_fullWorkAmount, nthr, ithr, start, end);

            std::vector<size_t> counters(dims_out.size(), 0);

            for (size_t iwork = start; iwork < end; ++iwork) {
                std::vector<T> src_f(this->_inputNum);
                T* dst_ptr_f = nullptr;
                this->init_ptr(args_ptrs, dims_out, counters, iwork, src_f, dst_ptr_f);

                switch (this->_opData.algo) {
                case Algorithm::EltwiseRelu:
                case Algorithm::EltwiseGeluErf:
                case Algorithm::EltwiseGeluTanh:
                case Algorithm::EltwiseElu:
                case Algorithm::EltwiseTanh:
                case Algorithm::EltwiseSigmoid:
                case Algorithm::EltwiseAbs:
                case Algorithm::EltwiseSqrt:
                case Algorithm::EltwiseSoftRelu:
                case Algorithm::EltwiseClamp:
                case Algorithm::EltwiseSwish:
                case Algorithm::EltwiseHswish:
                case Algorithm::EltwiseMish:
                case Algorithm::EltwiseHsigmoid:
                case Algorithm::EltwiseRoundHalfToEven:
                case Algorithm::EltwiseRoundHalfAwayFromZero:
                    *dst_ptr_f = ref_eltwise_injector->compute_scalar(src_f[0]);
                    break;
                case Algorithm::EltwiseAdd:
                    *dst_ptr_f = src_f[0] + src_f[1];
                    break;
                case Algorithm::EltwiseMulAdd:
                    *dst_ptr_f = src_f[0] * src_f[1] + src_f[2];
                    break;
                case Algorithm::EltwiseSubtract:
                    *dst_ptr_f = src_f[0] - src_f[1];
                    break;
                case Algorithm::EltwiseMultiply:
                    *dst_ptr_f = src_f[0] * src_f[1];
                    break;
                case Algorithm::EltwiseDivide:
                    *dst_ptr_f = src_f[0] / src_f[1];
                    break;
                case Algorithm::EltwiseCeiling:
                    *dst_ptr_f = ceilf(src_f[0]);
                    break;
                case Algorithm::EltwiseFloor:
                    *dst_ptr_f = floorf(src_f[0]);
                    break;
                case Algorithm::EltwiseNegative:
                    *dst_ptr_f = -src_f[0];
                    break;
                case Algorithm::EltwiseFloorMod:
                    *dst_ptr_f = src_f[0] - floorf(src_f[0] / src_f[1]) * src_f[1];
                    break;
                case Algorithm::EltwiseMod:
                    *dst_ptr_f = src_f[0] - truncf(src_f[0] / src_f[1]) * src_f[1];
                    break;
                case Algorithm::EltwiseMaximum:
                    *dst_ptr_f = std::max(src_f[0], src_f[1]);
                    break;
                case Algorithm::EltwiseMinimum:
                    *dst_ptr_f = std::min(src_f[0], src_f[1]);
                    break;
                case Algorithm::EltwiseExp:
                    *dst_ptr_f = expf(src_f[0]);
                    break;
                case Algorithm::EltwiseSquaredDifference:
                    *dst_ptr_f = powf((src_f[0] - src_f[1]), 2.F);
                    break;
                case Algorithm::EltwisePowerDynamic:
                    *dst_ptr_f = powf(src_f[0], src_f[1]);
                    break;
                case Algorithm::EltwiseEqual:
                    *dst_ptr_f = src_f[0] == src_f[1];
                    break;
                case Algorithm::EltwiseNotEqual:
                    *dst_ptr_f = src_f[0] != src_f[1];
                    break;
                case Algorithm::EltwiseGreater:
                    *dst_ptr_f = src_f[0] > src_f[1];
                    break;
                case Algorithm::EltwiseGreaterEqual:
                    *dst_ptr_f = src_f[0] >= src_f[1];
                    break;
                case Algorithm::EltwiseLess:
                    *dst_ptr_f = src_f[0] < src_f[1];
                    break;
                case Algorithm::EltwiseLessEqual:
                    *dst_ptr_f = src_f[0] <= src_f[1];
                    break;
                case Algorithm::EltwiseLogicalAnd:
                    *dst_ptr_f = src_f[0] && src_f[1];
                    break;
                case Algorithm::EltwiseLogicalOr:
                    *dst_ptr_f = src_f[0] || src_f[1];
                    break;
                case Algorithm::EltwiseLogicalXor:
                    *dst_ptr_f = (src_f[0] || src_f[1]) - (src_f[0] && src_f[1]);
                    break;
                case Algorithm::EltwiseLogicalNot:
                    *dst_ptr_f = !src_f[0];
                    break;
                case Algorithm::EltwisePrelu:
                    *dst_ptr_f = src_f[0] > 0 ? src_f[0] : static_cast<T>(src_f[0] * src_f[1]);
                    break;
                case Algorithm::EltwiseErf:
                    *dst_ptr_f = std::erf(src_f[0]);
                    break;
                case Algorithm::EltwiseSoftSign:
                    *dst_ptr_f = src_f[0] / (1 + std::fabs(src_f[0]));
                    break;
                // @todo implement proper isinfinite for non-float precisions
                case Algorithm::EltwiseIsFinite:
                    *dst_ptr_f = std::isfinite(static_cast<float>(src_f[0]));
                    break;
                case Algorithm::EltwiseIsInf:
                    *dst_ptr_f = (this->_opData.alpha && (src_f[0] == -std::numeric_limits<T>::infinity())) ||
                                 (this->_opData.beta && (src_f[0] == std::numeric_limits<T>::infinity()));
                    break;
                case Algorithm::EltwiseIsNaN:
                    *dst_ptr_f = std::isnan(src_f[0]);
                    break;
                case Algorithm::EltwiseSelect:
                    *dst_ptr_f = src_f[0] ? src_f[1] : src_f[2];
                    break;
                default:
                    OPENVINO_THROW("Unsupported operation type for Eltwise executor");
                }
            }
        });
    }
};

template <typename T,
          std::enable_if_t<std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int16_t> ||
                           std::is_same_v<T, uint16_t> || std::is_same_v<T, int32_t>>* = nullptr>
class BitwiseRefExecutor : public EltwiseRefBaseExecutor<T> {
public:
    BitwiseRefExecutor(const EltwiseData& opData, const VectorDims& outBlkDims, const std::vector<VectorDims>& inpDims)
        : EltwiseRefBaseExecutor<T>(opData, outBlkDims, inpDims) {}

    void exec(const jit_eltwise_call_args_ptrs& args_ptrs, const VectorDims& dims_out) override {
        std::shared_ptr<ref_eltwise_scalar_fwd_t> ref_eltwise_injector = nullptr;
        if (this->_opData.onednnAlgorithm != dnnl::algorithm::undef) {
            ref_eltwise_injector =
                std::make_shared<ref_eltwise_scalar_fwd_t>(static_cast<dnnl_alg_kind_t>(this->_opData.onednnAlgorithm),
                                                           this->_opData.alpha,
                                                           this->_opData.beta,
                                                           1.F);
        }

        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0;
            size_t end = 0;
            splitter(this->_fullWorkAmount, nthr, ithr, start, end);

            std::vector<size_t> counters(dims_out.size(), 0);

            for (size_t iwork = start; iwork < end; ++iwork) {
                std::vector<T> src_f(this->_inputNum);
                T* dst_ptr_f = nullptr;
                this->init_ptr(args_ptrs, dims_out, counters, iwork, src_f, dst_ptr_f);

                switch (this->_opData.algo) {
                case Algorithm::EltwiseBitwiseAnd: {
                    *dst_ptr_f = src_f[0] & src_f[1];
                    break;
                }
                case Algorithm::EltwiseBitwiseNot: {
                    *dst_ptr_f = ~src_f[0];
                    break;
                }
                case Algorithm::EltwiseBitwiseOr: {
                    *dst_ptr_f = src_f[0] | src_f[1];
                    break;
                }
                case Algorithm::EltwiseBitwiseXor: {
                    *dst_ptr_f = src_f[0] ^ src_f[1];
                    break;
                }
                case Algorithm::EltwiseBitwiseLeftShift: {
                    *dst_ptr_f = src_f[0] << src_f[1];
                    break;
                }
                case Algorithm::EltwiseBitwiseRightShift: {
                    *dst_ptr_f = src_f[0] >> src_f[1];
                    break;
                }
                default:
                    OPENVINO_THROW("Unsupported operation type for Eltwise executor");
                }
            }
        });
    }
};

}  // namespace

static Eltwise::executorPtr buildRefExecutor(const EltwiseKey& key) {
    switch (key.outPrc) {
    case ov::element::f16:
        return std::make_shared<EltwiseRefExecutor<dnnl::impl::float16_t>>(key.eltwise_data.front(),
                                                                           key.outBlkDims,
                                                                           key.inpDims);
    case ov::element::i8:
        return std::make_shared<BitwiseRefExecutor<element_type_traits<ov::element::i8>::value_type>>(
            key.eltwise_data.front(),
            key.outBlkDims,
            key.inpDims);

    case ov::element::u8:
        return std::make_shared<BitwiseRefExecutor<element_type_traits<ov::element::u8>::value_type>>(
            key.eltwise_data.front(),
            key.outBlkDims,
            key.inpDims);

    case ov::element::i16:
        return std::make_shared<BitwiseRefExecutor<element_type_traits<ov::element::i16>::value_type>>(
            key.eltwise_data.front(),
            key.outBlkDims,
            key.inpDims);

    case ov::element::u16:
        return std::make_shared<BitwiseRefExecutor<element_type_traits<ov::element::u16>::value_type>>(
            key.eltwise_data.front(),
            key.outBlkDims,
            key.inpDims);
    case ov::element::i32:
        return std::make_shared<BitwiseRefExecutor<element_type_traits<ov::element::i32>::value_type>>(
            key.eltwise_data.front(),
            key.outBlkDims,
            key.inpDims);

    default:
        // use float reference executor for any other precision for now
        return std::make_shared<EltwiseRefExecutor<float>>(key.eltwise_data.front(), key.outBlkDims, key.inpDims);
    }
}

static Eltwise::executorPtr buildExecutor(const EltwiseKey& key) {
    if (key.implType == EltwiseImplType::reference) {
        return buildRefExecutor(key);
    }

    return std::make_shared<EltwiseJitExecutor>(key.eltwise_data,
                                                key.ops_list,
                                                key.outBlkDims,
                                                key.outOrder,
                                                key.inpDims,
                                                key.inpPrc,
                                                key.outPrc,
                                                key.postOps,
                                                key.implType == EltwiseImplType::optimizedShapeAgnostic);
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
