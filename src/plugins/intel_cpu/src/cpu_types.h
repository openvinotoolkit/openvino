// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "utils/caseless.hpp"

namespace ov::intel_cpu {

using Dim = std::size_t;
using VectorDims = std::vector<Dim>;

std::string dim2str(Dim dim);
std::string dims2str(const VectorDims& dims);

enum class Type : uint8_t {
    Unknown,
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
    ROIAlignRotated,
    ROIPooling,
    PSROIPooling,
    BatchToSpace,
    DepthToSpace,
    Pad,
    Transpose,
    SpaceToBatch,
    SpaceToDepth,
    SparseFillEmptyRows,
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
    Col2Im,
    MVN,
    NormalizeL2,
    ScatterUpdate,
    ScatterElementsUpdate,
    ScatterNDUpdate,
    StringTensorPack,
    StringTensorUnpack,
    Interpolate,
    Reduce,
    Broadcast,
    EmbeddingBagPacked,
    EmbeddingBagOffsets,
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
    STFT,
    ISTFT,
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
    ExtractImagePatches,
    GenerateProposals,
    Inverse,
    NonMaxSuppression,
    MatrixNms,
    MulticlassNms,
    Multinomial,
    Subgraph,
    SubModel,
    PriorBox,
    PriorBoxClustered,
    Interaction,
    RandomUniform,
    Unique,
    Ngram,
    ScaledDotProductAttention,
    PagedAttention,
    RoPE,
    CausalMaskPreprocess,
    LLMMLP,
    QKVProjection,
    RMS,
    SearchSorted,
    SegmentMax,
    LoRA
};

enum class Algorithm : uint8_t {
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
    EltwiseFloor,
    EltwiseCeiling,
    EltwiseFloorMod,
    EltwiseNegative,
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
    EltwiseBitwiseLeftShift,
    EltwiseBitwiseRightShift,

    // FullyConnected algorithms
    FullyConnectedCommon,
    FullyConnectedCompressed,
    FullyConnectedQuantized,
    FullyConnectedQuantizedLegacy,

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

extern const ov::intel_cpu::caseless_unordered_map<std::string, Type> type_to_name_tbl;

Type TypeFromName(const std::string& type);

std::string NameFromType(Type type);
inline std::ostream& operator<<(std::ostream& os, const Type& type) {
    os << NameFromType(type);
    return os;
}

std::string algToString(Algorithm alg);

}  // namespace ov::intel_cpu
