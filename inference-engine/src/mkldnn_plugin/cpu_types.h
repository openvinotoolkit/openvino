// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace MKLDNNPlugin {

enum Algorithm {
    Undefined,

    // Pooling algorithms
    PoolingMax,
    PoolingAvg,

    // Convolution algorithms
    ConvolutionCommon,
    ConvolutionGrouped,

    // Convolution algorithms
    DeconvolutionCommon,
    DeconvolutionGrouped,

    // Elementwise algorithms
    EltwiseAdd,
    EltwiseMultiply,
    EltwiseSubtract,
    EltwiseDivide,
    EltwiseFloorMod,
    EltwiseMod,
    EltwiseMaximum,
    EltwiseMinimum,
    EltwiseSquaredDifference,
    EltwisePowerDynamic,
    EltwisePowerStatic,
    EltwiseMulAdd,
    EltwiseEqual,
    EltwiseNotEqual,
    EltwiseGreater,
    EltwiseGreaterEqual,
    EltwiseLess,
    EltwiseLessEqual,
    EltwiseLogicalAnd,
    EltwiseLogicalOr,
    EltwiseLogicalXor,
    EltwiseLogicalNot,
    EltwiseRelu,
    EltwiseGelu,
    EltwiseElu,
    EltwiseTanh,
    EltwiseSigmoid,
    EltwiseSquare, // TODO [NM]: looks like unused - remove
    EltwiseAbs,
    EltwiseSqrt,
    EltwiseLinear, // TODO [NM]: looks like unused - remove
    EltwiseBoundedRelu, // TODO [NM]: looks like unused - remove
    EltwiseSoftRelu, // TODO [NM]: looks like unused - remove
    EltwiseRelu6, // TODO [NM]: looks like unused - remove
    EltwiseExp,
    EltwiseClamp,
    EltwiseSwish,
    EltwisePrelu,
    EltwiseMish,
    EltwiseHswish,
    EltwiseHsigmoid,
    EltwiseRoundHalfToEven,
    EltwiseRoundHalfAwayFromZero,
    EltwiseErf,

    // FakeQuantize algorithms
    FQCommon,
    FQQuantization,
    FQBinarization,

    // ROIPooling algorithms
    ROIPoolingMax,
    ROIPoolingBilinear,

    // ROIAlign algorithms
    ROIAlignMax,
    ROIAlignAvg,

    // PSROIPooling algorithms
    PSROIPoolingAverage,
    PSROIPoolingBilinear,
    PSROIPoolingBilinearDeformable,

    // Reduce algorithms
    ReduceL1,
    ReduceL2,
    ReduceAnd,
    ReduceOr,
    ReduceMax,
    ReduceMean,
    ReduceMin,
    ReduceProd,
    ReduceSum,
    ReduceLogSum,
    ReduceLogSumExp,
    ReduceSumSquare,

    // Math algorithms
    MathAbs,
    MathAcos,
    MathAcosh,
    MathAsin,
    MathAsinh,
    MathAtan,
    MathAtanh,
    MathCeiling,
    MathCos,
    MathCosh,
    MathErf,
    MathFloor,
    MathHardSigmoid,
    MathLog,
    MathNegative,
    MathReciprocal,
    MathSelu,
    MathSign,
    MathSin,
    MathSinh,
    MathSoftPlus,
    MathSoftsign,
    MathTan,

    // Split algorithms
    SplitDefault,
    SplitVariadic
};

} // namespace MKLDNNPlugin
