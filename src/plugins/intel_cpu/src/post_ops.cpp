// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "post_ops.hpp"

#include "node.h"
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
        return EltwiseKind::Activation;
    case Algorithm::EltwiseAdd:
    case Algorithm::EltwiseSubtract:
    case Algorithm::EltwiseDivide:
    case Algorithm::EltwiseMultiply:
    case Algorithm::EltwiseMulAdd:
    case Algorithm::EltwisePowerStatic:
    case Algorithm::EltwisePrelu:
        return EltwiseKind::ScaleShift;
    default:
        OPENVINO_THROW("Unexpected eltwise algorithm: ", algToString(alg));
    }
}

ScaleShiftPostOp::Type convertToScaleShiftOpt(const Algorithm alg) {
    switch (alg) {
    case Algorithm::EltwiseAdd:
        return ScaleShiftPostOp::add;
    case Algorithm::EltwiseSubtract:
        return ScaleShiftPostOp::subtract;
    case Algorithm::EltwiseDivide:
        return ScaleShiftPostOp::divide;
    case Algorithm::EltwiseMultiply:
        return ScaleShiftPostOp::multiply;
    case Algorithm::EltwiseMulAdd:
        return ScaleShiftPostOp::muladd;
    case Algorithm::EltwisePowerStatic:
        return ScaleShiftPostOp::powerstatic;
    case Algorithm::EltwisePrelu:
        return ScaleShiftPostOp::prelu;
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

    for (const auto& node : fused) {
        if (const auto eltwise = std::dynamic_pointer_cast<node::Eltwise>(node)) {
            const auto eltwiseKind = getEltwiseKind(eltwise->getAlgorithm());
            switch (eltwiseKind) {
            case EltwiseKind::Activation:
                ops.push_back(makeActivationPostOp(eltwise));
                break;
            case EltwiseKind::ScaleShift:
                ops.push_back(makeScaleShiftPostOp(eltwise));
                break;
            }
        }

        if (const auto fq = std::dynamic_pointer_cast<node::FakeQuantize>(node)) {
            ops.push_back(std::make_shared<FakeQuantizePostOp>(fq->getCropLow(),
                                                               fq->getCropHigh(),
                                                               fq->getInputScale(),
                                                               fq->getInputShift(),
                                                               fq->getOutputScale(),
                                                               fq->getOutputShift(),
                                                               fq->getLevels()));
        }
    }

    return ops;
}

}  // namespace ov::intel_cpu
