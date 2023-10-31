// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "caseless.hpp"

namespace ov {
namespace intel_cpu {

using Dim = std::size_t;
using VectorDims = std::vector<Dim>;

enum class Type {
    Unknown,
    Generic,
    If,
    Reorder,
    Input,
    Output,
    Eye,
    Convolution,
    Deconvolution,
    Lrn,
    Pooling,
    AdaptivePooling,
    FullyConnected,
    Softmax,
    Split,
    Concatenation,
    Eltwise,
    MatMul,
    Reshape,
    ShapeOf,
    NonZero,
    Tile,
    ROIAlign,
    ROIPooling,
    PSROIPooling,
    BatchToSpace,
    DepthToSpace,
    Pad,
    Transpose,
    SpaceToBatch,
    SpaceToDepth,
    StridedSlice,
    MemoryOutput,
    MemoryInput,
    RNNCell,
    RNNSeq,
    FakeQuantize,
    BinaryConvolution,
    DeformableConvolution,
    TensorIterator,
    Convert,
    ColorConvert,
    MVN,
    NormalizeL2,
    ScatterUpdate,
    ScatterElementsUpdate,
    ScatterNDUpdate,
    Interpolate,
    Reduce,
    Broadcast,
    EmbeddingSegmentsSum,
    EmbeddingBagPackedSum,
    EmbeddingBagOffsetsSum,
    Gather,
    GatherElements,
    GatherND,
    GridSample,
    OneHot,
    RegionYolo,
    Roll,
    Reference,
    ShuffleChannels,
    DFT,
    RDFT,
    Math,
    CTCLoss,
    Bucketize,
    CTCGreedyDecoder,
    CTCGreedyDecoderSeqLen,
    CumSum,
    DetectionOutput,
    ExperimentalDetectronDetectionOutput,
    LogSoftmax,
    TopK,
    GatherTree,
    GRN,
    Range,
    Proposal,
    ReorgYolo,
    ReverseSequence,
    ExperimentalDetectronTopKROIs,
    ExperimentalDetectronROIFeatureExtractor,
    ExperimentalDetectronPriorGridGenerator,
    ExperimentalDetectronGenerateProposalsSingleImage,
    GenerateProposals,
    ExtractImagePatches,
    NonMaxSuppression,
    MatrixNms,
    MulticlassNms,
    Multinomial,
    Subgraph,
    PriorBox,
    PriorBoxClustered,
    Interaction,
    MHA,
    RandomUniform,
    Unique,
    Ngram
};

enum class Algorithm {
    Default,

    // Pooling algorithms
    PoolingMax,
    PoolingAvg,

    // Adaptive pooling algorithms
    AdaptivePoolingMax,
    AdaptivePoolingAvg,

    // Convolution algorithms
    ConvolutionCommon,
    ConvolutionGrouped,

    // Convolution algorithms
    DeconvolutionCommon,
    DeconvolutionGrouped,

    // Elementwise algorithms
    EltwiseAdd,
    EltwiseIsFinite,
    EltwiseIsInf,
    EltwiseIsNaN,
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
    EltwiseGeluErf,
    EltwiseGeluTanh,
    EltwiseElu,
    EltwiseTanh,
    EltwiseSigmoid,
    EltwiseAbs,
    EltwiseSelect,
    EltwiseSqrt,
    EltwiseSoftRelu,
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
    EltwiseSoftSign,
    EltwiseLog,
    EltwiseBitwiseAnd,
    EltwiseBitwiseNot,
    EltwiseBitwiseOr,
    EltwiseBitwiseXor,

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
    MathNegative,
    MathReciprocal,
    MathSelu,
    MathSign,
    MathSin,
    MathSinh,
    MathSoftPlus,
    MathSoftsign,
    MathTan,
    // TensorIterator
    TensorIteratorCommon,
    TensorIteratorLoop,
    // Color conversions
    ColorConvertNV12toRGB,
    ColorConvertNV12toBGR,
    ColorConvertI420toRGB,
    ColorConvertI420toBGR,
};

extern const InferenceEngine::details::caseless_unordered_map<std::string, Type> type_to_name_tbl;

Type TypeFromName(const std::string& type);

std::string NameFromType(const Type type);

std::string algToString(const Algorithm alg);

}  // namespace intel_cpu
}  // namespace ov
