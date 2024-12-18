// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "post_ops.hpp"

#include <cstddef>

#include "cpu_types.h"
#include "node.h"
#include "nodes/conv.h"
#include "nodes/eltwise.h"
#include "nodes/fake_quantize.h"

namespace ov::intel_cpu {

EltwiseKind getEltwiseKind(const Algorithm alg) {
    switch (alg) {
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
        return EltwiseKind::Activation;
    case Algorithm::EltwiseAdd:
    case Algorithm::EltwiseSubtract:
    case Algorithm::EltwiseDivide:
    case Algorithm::EltwiseMultiply:
    case Algorithm::EltwiseMulAdd:
    case Algorithm::EltwisePrelu:
        return EltwiseKind::ScaleShift;
    default:
        OPENVINO_THROW("Unexpected eltwise algorithm: ", algToString(alg));
    }
}

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
    case Algorithm::EltwisePrelu:
        return ScaleShiftPostOp::Type::prelu;
    default:
        OPENVINO_THROW("Unexpected eltwise algorithm: ", algToString(alg));
    }
}

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
    case ActivationPostOp::Type::square:
        OPENVINO_THROW("square is not supported");
    case ActivationPostOp::Type::linear:
        OPENVINO_THROW("linear is not supported");
    }

    OPENVINO_THROW("Unsupported algorithm");
}

PostOps getPostOps(const std::vector<NodePtr>& fused) {
    PostOps ops;

    auto makeActivationPostOp = [](const std::shared_ptr<node::Eltwise>& eltwise) {
        return std::make_shared<ActivationPostOp>(convertToActivationPostOpt(eltwise->getAlgorithm()),
                                                  eltwise->getAlpha(),
                                                  eltwise->getBeta(),
                                                  eltwise->getGamma());
    };

    auto makeScaleShiftPostOp = [](const std::shared_ptr<node::Eltwise>& eltwise) {
        return std::make_shared<ScaleShiftPostOp>(convertToScaleShiftOpt(eltwise->getAlgorithm()),
                                                  eltwise->getScales(),
                                                  eltwise->getShifts());
    };

    auto makeSumPostOp = [](const std::shared_ptr<node::Eltwise>& eltwise) {
        return std::make_shared<SumPostOp>(1.f, 0);
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
            ops.push_back(std::make_shared<FakeQuantizePostOp>(convertToFqPostOp(fq->getAlgorithm()),
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

            ops.push_back(std::make_shared<DepthwiseConvolutionPostOp>(ih, iw, kernel, strides));
        }
    }

    return ops;
}

}  // namespace ov::intel_cpu
