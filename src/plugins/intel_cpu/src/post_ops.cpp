// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "post_ops.hpp"

#include <any>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "cpu_types.h"
#include "node.h"
#include "nodes/conv.h"
#include "nodes/eltwise.h"
#include "nodes/fake_quantize.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

// classify all eltwise algorithms
EltwiseKind getEltwiseKind(const Algorithm alg) {
    switch (alg) {
    // Activation algorithms
    case Algorithm::EltwiseSqrt:
    case Algorithm::EltwiseRelu:
    case Algorithm::EltwiseTanh:
    case Algorithm::EltwiseElu:
    case Algorithm::EltwiseAbs:
    case Algorithm::EltwiseSoftRelu:
    case Algorithm::EltwiseSigmoid:
    case Algorithm::EltwiseExp:
    case Algorithm::EltwiseGeluErf:
    case Algorithm::EltwiseGeluTanh:
    case Algorithm::EltwiseClamp:
    case Algorithm::EltwiseSwish:
    case Algorithm::EltwiseHswish:
    case Algorithm::EltwiseMish:
    case Algorithm::EltwiseHsigmoid:
    case Algorithm::EltwiseRoundHalfToEven:
    case Algorithm::EltwiseRoundHalfAwayFromZero:
    case Algorithm::EltwisePowerStatic:
    case Algorithm::EltwiseFloor:
    case Algorithm::EltwiseCeiling:
    case Algorithm::EltwiseNegative:
    case Algorithm::EltwiseErf:
    case Algorithm::EltwiseSoftSign:
    case Algorithm::EltwiseLog:
        return EltwiseKind::Activation;
    // ScaleShift algorithms
    case Algorithm::EltwiseAdd:
    case Algorithm::EltwiseSubtract:
    case Algorithm::EltwiseDivide:
    case Algorithm::EltwiseMultiply:
    case Algorithm::EltwiseMulAdd:
    case Algorithm::EltwisePowerDynamic:
    case Algorithm::EltwisePrelu:
    case Algorithm::EltwiseSelect:
    case Algorithm::EltwiseMaximum:
    case Algorithm::EltwiseMinimum:
    case Algorithm::EltwiseSquaredDifference:
    case Algorithm::EltwiseIsFinite:
    case Algorithm::EltwiseIsInf:
    case Algorithm::EltwiseIsNaN:
    case Algorithm::EltwiseEqual:
    case Algorithm::EltwiseNotEqual:
    case Algorithm::EltwiseGreater:
    case Algorithm::EltwiseGreaterEqual:
    case Algorithm::EltwiseLess:
    case Algorithm::EltwiseLessEqual:
    case Algorithm::EltwiseLogicalAnd:
    case Algorithm::EltwiseLogicalOr:
    case Algorithm::EltwiseLogicalXor:
    case Algorithm::EltwiseLogicalNot:
    case Algorithm::EltwiseFloorMod:
    case Algorithm::EltwiseMod:
    case Algorithm::EltwiseBitwiseAnd:
    case Algorithm::EltwiseBitwiseNot:
    case Algorithm::EltwiseBitwiseOr:
    case Algorithm::EltwiseBitwiseXor:
    case Algorithm::EltwiseBitwiseLeftShift:
    case Algorithm::EltwiseBitwiseRightShift:
        return EltwiseKind::ScaleShift;

    default:
        OPENVINO_THROW("Unexpected eltwise algorithm: ", algToString(alg));
    }
}

// Convert Algorithm to ScaleShiftPostOp::Type
ScaleShiftPostOp::Type convertToScaleShiftOpt(const Algorithm alg) {
    switch (alg) {
    case Algorithm::EltwiseAdd:
        return ScaleShiftPostOp::Type::add;
    case Algorithm::EltwiseSubtract:
        return ScaleShiftPostOp::Type::subtract;
    case Algorithm::EltwiseDivide:
        return ScaleShiftPostOp::Type::divide;
    case Algorithm::EltwiseMultiply:
        return ScaleShiftPostOp::Type::multiply;
    case Algorithm::EltwiseMulAdd:
        return ScaleShiftPostOp::Type::muladd;
    case Algorithm::EltwisePowerDynamic:
        return ScaleShiftPostOp::Type::power_dynamic;
    case Algorithm::EltwisePrelu:
        return ScaleShiftPostOp::Type::prelu;
    case Algorithm::EltwiseSelect:
        return ScaleShiftPostOp::Type::select;
    case Algorithm::EltwiseMaximum:
        return ScaleShiftPostOp::Type::maximum;
    case Algorithm::EltwiseMinimum:
        return ScaleShiftPostOp::Type::minimum;
    case Algorithm::EltwiseSquaredDifference:
        return ScaleShiftPostOp::Type::squared_difference;
    case Algorithm::EltwiseLogicalAnd:
        return ScaleShiftPostOp::Type::logical_and;
    case Algorithm::EltwiseLogicalOr:
        return ScaleShiftPostOp::Type::logical_or;
    case Algorithm::EltwiseLogicalXor:
        return ScaleShiftPostOp::Type::logical_xor;
    case Algorithm::EltwiseLogicalNot:
        return ScaleShiftPostOp::Type::logical_not;
    case Algorithm::EltwiseFloorMod:
        return ScaleShiftPostOp::Type::floor_mod;
    case Algorithm::EltwiseMod:
        return ScaleShiftPostOp::Type::mod;
    case Algorithm::EltwiseEqual:
        return ScaleShiftPostOp::Type::equal;
    case Algorithm::EltwiseNotEqual:
        return ScaleShiftPostOp::Type::not_equal;
    case Algorithm::EltwiseGreater:
        return ScaleShiftPostOp::Type::greater;
    case Algorithm::EltwiseGreaterEqual:
        return ScaleShiftPostOp::Type::greater_equal;
    case Algorithm::EltwiseLess:
        return ScaleShiftPostOp::Type::less;
    case Algorithm::EltwiseLessEqual:
        return ScaleShiftPostOp::Type::less_equal;
    case Algorithm::EltwiseIsFinite:
        return ScaleShiftPostOp::Type::is_finite;
    case Algorithm::EltwiseIsInf:
        return ScaleShiftPostOp::Type::is_inf;
    case Algorithm::EltwiseIsNaN:
        return ScaleShiftPostOp::Type::is_nan;
    case Algorithm::EltwiseBitwiseAnd:
        return ScaleShiftPostOp::Type::bitwise_and;
    case Algorithm::EltwiseBitwiseNot:
        return ScaleShiftPostOp::Type::bitwise_not;
    case Algorithm::EltwiseBitwiseOr:
        return ScaleShiftPostOp::Type::bitwise_or;
    case Algorithm::EltwiseBitwiseXor:
        return ScaleShiftPostOp::Type::bitwise_xor;
    default:
        OPENVINO_THROW("Unexpected eltwise algorithm: ", algToString(alg));
    }
}

// Convert Algorithm to ActivationPostOp::Type
ActivationPostOp::Type convertToActivationPostOpt(const Algorithm alg) {
    switch (alg) {
    case Algorithm::EltwiseSqrt:
        return ActivationPostOp::Type::sqrt;
    case Algorithm::EltwiseRelu:
        return ActivationPostOp::Type::relu;
    case Algorithm::EltwiseTanh:
        return ActivationPostOp::Type::tanh;
    case Algorithm::EltwiseElu:
        return ActivationPostOp::Type::elu;
    case Algorithm::EltwiseAbs:
        return ActivationPostOp::Type::abs;
    case Algorithm::EltwiseSoftRelu:
        return ActivationPostOp::Type::soft_relu;
    case Algorithm::EltwiseSigmoid:
        return ActivationPostOp::Type::logistic;
    case Algorithm::EltwiseExp:
        return ActivationPostOp::Type::exp;
    case Algorithm::EltwiseGeluErf:
        return ActivationPostOp::Type::gelu_erf;
    case Algorithm::EltwiseGeluTanh:
        return ActivationPostOp::Type::gelu_tanh;
    case Algorithm::EltwiseClamp:
        return ActivationPostOp::Type::clip;
    case Algorithm::EltwiseSwish:
        return ActivationPostOp::Type::swish;
    case Algorithm::EltwiseHswish:
        return ActivationPostOp::Type::hardswish;
    case Algorithm::EltwiseMish:
        return ActivationPostOp::Type::mish;
    case Algorithm::EltwiseHsigmoid:
        return ActivationPostOp::Type::hsigmoid;
    case Algorithm::EltwiseRoundHalfToEven:
        return ActivationPostOp::Type::round_half_to_even;
    case Algorithm::EltwiseRoundHalfAwayFromZero:
        return ActivationPostOp::Type::round_half_away_from_zero;
    case Algorithm::EltwisePowerStatic:
        return ActivationPostOp::Type::powerstatic;
    case Algorithm::EltwiseFloor:
        return ActivationPostOp::Type::floor;
    case Algorithm::EltwiseCeiling:
        return ActivationPostOp::Type::ceiling;
    case Algorithm::EltwiseNegative:
        return ActivationPostOp::Type::negative;
    case Algorithm::EltwiseErf:
        return ActivationPostOp::Type::erf;
    case Algorithm::EltwiseSoftSign:
        return ActivationPostOp::Type::soft_sign;
    case Algorithm::EltwiseLog:
        return ActivationPostOp::Type::log;
    default:
        OPENVINO_THROW("Unexpected eltwise algorithm: ", algToString(alg));
    }
}

FakeQuantizePostOp::Type convertToFqPostOp(const Algorithm alg) {
    switch (alg) {
    case ov::intel_cpu::Algorithm::FQBinarization:
        return FakeQuantizePostOp::Type::binarization;
    case ov::intel_cpu::Algorithm::FQQuantization:
        return FakeQuantizePostOp::Type::quantization_only;
    case ov::intel_cpu::Algorithm::FQCommon:
        return FakeQuantizePostOp::Type::quantization_dequantization;
    default:
        OPENVINO_THROW("Unexpected eltwise algorithm: ", algToString(alg));
    }
}

Algorithm convertToEltwiseAlgorithm(const ActivationPostOp::Type type) {
    switch (type) {
    case ActivationPostOp::Type::sqrt:
        return Algorithm::EltwiseSqrt;
    case ActivationPostOp::Type::relu:
        return Algorithm::EltwiseRelu;
    case ActivationPostOp::Type::tanh:
        return Algorithm::EltwiseTanh;
    case ActivationPostOp::Type::elu:
        return Algorithm::EltwiseElu;
    case ActivationPostOp::Type::abs:
        return Algorithm::EltwiseAbs;
    case ActivationPostOp::Type::soft_relu:
        return Algorithm::EltwiseSoftRelu;
    case ActivationPostOp::Type::logistic:
        return Algorithm::EltwiseSigmoid;
    case ActivationPostOp::Type::exp:
        return Algorithm::EltwiseExp;
    case ActivationPostOp::Type::gelu_erf:
        return Algorithm::EltwiseGeluErf;
    case ActivationPostOp::Type::gelu_tanh:
        return Algorithm::EltwiseGeluTanh;
    case ActivationPostOp::Type::clip:
        return Algorithm::EltwiseClamp;
    case ActivationPostOp::Type::swish:
        return Algorithm::EltwiseSwish;
    case ActivationPostOp::Type::hardswish:
        return Algorithm::EltwiseHswish;
    case ActivationPostOp::Type::mish:
        return Algorithm::EltwiseMish;
    case ActivationPostOp::Type::hsigmoid:
        return Algorithm::EltwiseHsigmoid;
    case ActivationPostOp::Type::round_half_to_even:
        return Algorithm::EltwiseRoundHalfToEven;
    case ActivationPostOp::Type::round_half_away_from_zero:
        return Algorithm::EltwiseRoundHalfAwayFromZero;
    case ActivationPostOp::Type::powerstatic:
        return Algorithm::EltwisePowerStatic;
    case ActivationPostOp::Type::floor:
        return Algorithm::EltwiseFloor;
    case ActivationPostOp::Type::ceiling:
        return Algorithm::EltwiseCeiling;
    case ActivationPostOp::Type::negative:
        return Algorithm::EltwiseNegative;
    case ActivationPostOp::Type::erf:
        return Algorithm::EltwiseErf;
    case ActivationPostOp::Type::soft_sign:
        return Algorithm::EltwiseSoftSign;
    case ActivationPostOp::Type::log:
        return Algorithm::EltwiseLog;
    default:
        OPENVINO_THROW("Unsupported ActivationPostOp::Type");
    }
}

Algorithm convertToEltwiseAlgorithm(const ScaleShiftPostOp::Type type) {
    switch (type) {
    case ScaleShiftPostOp::Type::add:
        return Algorithm::EltwiseAdd;
    case ScaleShiftPostOp::Type::subtract:
        return Algorithm::EltwiseSubtract;
    case ScaleShiftPostOp::Type::divide:
        return Algorithm::EltwiseDivide;
    case ScaleShiftPostOp::Type::multiply:
        return Algorithm::EltwiseMultiply;
    case ScaleShiftPostOp::Type::muladd:
        return Algorithm::EltwiseMulAdd;
    case ScaleShiftPostOp::Type::power_dynamic:
        return Algorithm::EltwisePowerDynamic;
    case ScaleShiftPostOp::Type::prelu:
        return Algorithm::EltwisePrelu;
    case ScaleShiftPostOp::Type::select:
        return Algorithm::EltwiseSelect;
    case ScaleShiftPostOp::Type::maximum:
        return Algorithm::EltwiseMaximum;
    case ScaleShiftPostOp::Type::minimum:
        return Algorithm::EltwiseMinimum;
    case ScaleShiftPostOp::Type::squared_difference:
        return Algorithm::EltwiseSquaredDifference;
    case ScaleShiftPostOp::Type::logical_and:
        return Algorithm::EltwiseLogicalAnd;
    case ScaleShiftPostOp::Type::logical_or:
        return Algorithm::EltwiseLogicalOr;
    case ScaleShiftPostOp::Type::logical_xor:
        return Algorithm::EltwiseLogicalXor;
    case ScaleShiftPostOp::Type::logical_not:
        return Algorithm::EltwiseLogicalNot;
    case ScaleShiftPostOp::Type::floor_mod:
        return Algorithm::EltwiseFloorMod;
    case ScaleShiftPostOp::Type::mod:
        return Algorithm::EltwiseMod;
    case ScaleShiftPostOp::Type::equal:
        return Algorithm::EltwiseEqual;
    case ScaleShiftPostOp::Type::not_equal:
        return Algorithm::EltwiseNotEqual;
    case ScaleShiftPostOp::Type::greater:
        return Algorithm::EltwiseGreater;
    case ScaleShiftPostOp::Type::greater_equal:
        return Algorithm::EltwiseGreaterEqual;
    case ScaleShiftPostOp::Type::less:
        return Algorithm::EltwiseLess;
    case ScaleShiftPostOp::Type::less_equal:
        return Algorithm::EltwiseLessEqual;
    case ScaleShiftPostOp::Type::is_finite:
        return Algorithm::EltwiseIsFinite;
    case ScaleShiftPostOp::Type::is_inf:
        return Algorithm::EltwiseIsInf;
    case ScaleShiftPostOp::Type::is_nan:
        return Algorithm::EltwiseIsNaN;
    case ScaleShiftPostOp::Type::bitwise_and:
        return Algorithm::EltwiseBitwiseAnd;
    case ScaleShiftPostOp::Type::bitwise_not:
        return Algorithm::EltwiseBitwiseNot;
    case ScaleShiftPostOp::Type::bitwise_or:
        return Algorithm::EltwiseBitwiseOr;
    case ScaleShiftPostOp::Type::bitwise_xor:
        return Algorithm::EltwiseBitwiseXor;
    default:
        OPENVINO_THROW("Unsupported ScaleShiftPostOp::Type");
    }
}

dnnl::algorithm convertToDnnlAlgorithm(const ActivationPostOp::Type m_type) {
    switch (m_type) {
    case ActivationPostOp::Type::relu:
        return dnnl::algorithm::eltwise_relu;
    case ActivationPostOp::Type::gelu_erf:
        return dnnl::algorithm::eltwise_gelu_erf;
    case ActivationPostOp::Type::gelu_tanh:
        return dnnl::algorithm::eltwise_gelu_tanh;
    case ActivationPostOp::Type::elu:
        return dnnl::algorithm::eltwise_elu;
    case ActivationPostOp::Type::tanh:
        return dnnl::algorithm::eltwise_tanh;
    case ActivationPostOp::Type::logistic:
        return dnnl::algorithm::eltwise_logistic;
    case ActivationPostOp::Type::abs:
        return dnnl::algorithm::eltwise_abs;
    case ActivationPostOp::Type::sqrt:
        return dnnl::algorithm::eltwise_sqrt;
    case ActivationPostOp::Type::clip:
        return dnnl::algorithm::eltwise_clip;
    case ActivationPostOp::Type::swish:
        return dnnl::algorithm::eltwise_swish;
    case ActivationPostOp::Type::hardswish:
        return dnnl::algorithm::eltwise_hardswish;
    case ActivationPostOp::Type::mish:
        return dnnl::algorithm::eltwise_mish;
    case ActivationPostOp::Type::hsigmoid:
        return dnnl::algorithm::eltwise_hsigmoid;
    case ActivationPostOp::Type::round_half_to_even:
        return dnnl::algorithm::eltwise_round_half_to_even;
    case ActivationPostOp::Type::round_half_away_from_zero:
        return dnnl::algorithm::eltwise_round_half_away_from_zero;
    case ActivationPostOp::Type::soft_relu:
        return dnnl::algorithm::eltwise_soft_relu;
    default:  // handle for all the cases can be added if necessary
        return dnnl::algorithm::undef;
    }
}

PostOps getPostOps(const std::vector<NodePtr>& fused, ov::element::Type_t sumDataType) {
    PostOps ops;

    auto makeActivationPostOp = [](const std::shared_ptr<node::Eltwise>& eltwise) {
        return std::make_any<ActivationPostOp>(convertToActivationPostOpt(eltwise->getAlgorithm()),
                                               eltwise->getAlpha(),
                                               eltwise->getBeta(),
                                               eltwise->getGamma());
    };

    auto makeScaleShiftPostOp = [](const std::shared_ptr<node::Eltwise>& eltwise) {
        return std::make_any<ScaleShiftPostOp>(convertToScaleShiftOpt(eltwise->getAlgorithm()),
                                               eltwise->getScales(),
                                               eltwise->getShifts());
    };

    auto makeSumPostOp = [&](const std::shared_ptr<node::Eltwise>& eltwise) {
        OPENVINO_ASSERT(sumDataType != ov::element::dynamic, "Sum data type is not defined ", eltwise->getName());
        return std::make_any<SumPostOp>(1.0, 0, sumDataType);
    };

    for (const auto& node : fused) {
        if (const auto eltwise = std::dynamic_pointer_cast<node::Eltwise>(node)) {
            const auto eltwiseKind = getEltwiseKind(eltwise->getAlgorithm());
            switch (eltwiseKind) {
            case EltwiseKind::Activation:
                ops.push_back(makeActivationPostOp(eltwise));
                break;
            case EltwiseKind::ScaleShift:
                if (eltwise->isSpecialConvolutionAddFusing()) {
                    ops.push_back(makeSumPostOp(eltwise));
                } else {
                    ops.push_back(makeScaleShiftPostOp(eltwise));
                }
                break;
            }
        }

        if (const auto fq = std::dynamic_pointer_cast<node::FakeQuantize>(node)) {
            ops.push_back(std::make_any<FakeQuantizePostOp>(convertToFqPostOp(fq->getAlgorithm()),
                                                            fq->getCropLow(),
                                                            fq->getCropHigh(),
                                                            fq->getInputScale(),
                                                            fq->getInputShift(),
                                                            fq->getOutputScale(),
                                                            fq->getOutputShift(),
                                                            fq->getLevels(),
                                                            fq->isInputLowBroadcast(),
                                                            fq->isOutputHighBroadcast()));
        }

        if (const auto conv = std::dynamic_pointer_cast<node::Convolution>(node)) {
            const auto& inputShape = conv->getInputShapeAtPort(0);
            const auto& inActivationDims = inputShape.getStaticDims();
            const size_t ih = inActivationDims[inputShape.getRank() - 2];
            const size_t iw = inActivationDims[inputShape.getRank() - 1];

            const auto& wieghtsShape = conv->getInputShapeAtPort(1);
            const auto& dwWeightsDims = wieghtsShape.getStaticDims();
            const std::vector<size_t> kernel{dwWeightsDims[dwWeightsDims.size() - 1],
                                             dwWeightsDims[dwWeightsDims.size() - 2]};
            const auto& strides = conv->getStride();

            ops.push_back(std::make_any<DepthwiseConvolutionPostOp>(ih, iw, kernel, strides));
        }
    }

    return ops;
}

}  // namespace ov::intel_cpu
