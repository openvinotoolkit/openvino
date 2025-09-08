// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov::op::v0 {
class Abs;
class Acos;
class Asin;
class Atan;
class BatchNormInference;
class CTCGreedyDecoder;
class Ceiling;
class Clamp;
class Concat;
class Constant;
class Convert;
class Cos;
class Cosh;
class CumSum;
class DepthToSpace;
class DetectionOutput;
class Elu;
class Erf;
class Exp;
class FakeQuantize;
class Floor;
class GRN;
class Gelu;
class HardSigmoid;
class Interpolate;
class LRN;
class LSTMCell;
class Log;
class MVN;
class MatMul;
class Negative;
class NormalizeL2;
class PRelu;
class PSROIPooling;
class Parameter;
class PriorBox;
class PriorBoxClustered;
class Proposal;
class RNNCell;
class ROIPooling;
class Range;
class RegionYolo;
class Relu;
class ReorgYolo;
class Result;
class ReverseSequence;
class Selu;
class ShapeOf;
class ShuffleChannels;
class Sigmoid;
class Sign;
class Sin;
class Sinh;
class SpaceToDepth;
class Sqrt;
class SquaredDifference;
class Squeeze;
class Tan;
class Tanh;
class TensorIterator;
class Tile;
class Unsqueeze;
class Xor;
}  // namespace ov::op::v0

namespace ov::op::v1 {
class Add;
class AvgPool;
class BatchToSpace;
class BinaryConvolution;
class Broadcast;
class ConvertLike;
class Convolution;
class ConvolutionBackpropData;
class DeformableConvolution;
class DeformablePSROIPooling;
class Divide;
class Equal;
class FloorMod;
class Gather;
class GatherTree;
class Greater;
class GreaterEqual;
class GroupConvolution;
class GroupConvolutionBackpropData;
class Less;
class LessEqual;
class LogicalAnd;
class LogicalNot;
class LogicalOr;
class LogicalXor;
class MaxPool;
class Maximum;
class Minimum;
class Mod;
class Multiply;
class NonMaxSuppression;
class NotEqual;
class OneHot;
class Pad;
class Power;
class ReduceLogicalAnd;
class ReduceLogicalOr;
class ReduceMax;
class ReduceMean;
class ReduceMin;
class ReduceProd;
class ReduceSum;
class Reshape;
class Reverse;
class Select;
class Softmax;
class SpaceToBatch;
class Split;
class StridedSlice;
class Subtract;
class TopK;
class Transpose;
class VariadicSplit;
}  // namespace ov::op::v1

namespace ov::op::v3 {
class Acosh;
class Asinh;
class Assign;
class Atanh;
class Broadcast;
class Bucketize;
class EmbeddingBagOffsetsSum;
class EmbeddingBagPackedSum;
class EmbeddingSegmentsSum;
class ExtractImagePatches;
class GRUCell;
class NonMaxSuppression;
class NonZero;
class ROIAlign;
class ReadValue;
class ScatterElementsUpdate;
class ScatterNDUpdate;
class ScatterUpdate;
class ShapeOf;
class TopK;
}  // namespace ov::op::v3

namespace ov::op::v4 {
class CTCLoss;
class HSwish;
class Interpolate;
class LSTMCell;
class Mish;
class NonMaxSuppression;
class Proposal;
class Range;
class ReduceL1;
class ReduceL2;
class SoftPlus;
class Swish;
}  // namespace ov::op::v4

namespace ov::op::v5 {
class BatchNormInference;
class GRUSequence;
class GatherND;
class HSigmoid;
class LSTMSequence;
class LogSoftmax;
class Loop;
class NonMaxSuppression;
class RNNSequence;
class Round;
}  // namespace ov::op::v5

namespace ov::op::v6 {
class Assign;
class CTCGreedyDecoderSeqLen;
class ExperimentalDetectronDetectionOutput;
class ExperimentalDetectronGenerateProposalsSingleImage;
class ExperimentalDetectronPriorGridGenerator;
class ExperimentalDetectronROIFeatureExtractor;
class ExperimentalDetectronTopKROIs;
class GatherElements;
class MVN;
class ReadValue;
}  // namespace ov::op::v6

namespace ov::op::v7 {
class DFT;
class Einsum;
class Gather;
class Gelu;
class IDFT;
class Roll;
}  // namespace ov::op::v7

namespace ov::op::v8 {
class AdaptiveAvgPool;
class AdaptiveMaxPool;
class DeformableConvolution;
class DetectionOutput;
class Gather;
class GatherND;
class I420toBGR;
class I420toRGB;
class If;
class MatrixNms;
class MaxPool;
class MulticlassNms;
class NV12toBGR;
class NV12toRGB;
class PriorBox;
class RandomUniform;
class Slice;
class Softmax;
}  // namespace ov::op::v8

namespace ov::op::v9 {
class Eye;
class GenerateProposals;
class GridSample;
class IRDFT;
class MulticlassNms;
class NonMaxSuppression;
class RDFT;
class ROIAlign;
class SoftSign;
}  // namespace ov::op::v9

namespace ov::op::v10 {
class IsFinite;
class IsInf;
class IsNaN;
class Unique;
}  // namespace ov::op::v10

namespace ov::op::v11 {
class Interpolate;
class TopK;
}  // namespace ov::op::v11

namespace ov::op::v12 {
class GroupNormalization;
class Pad;
class ScatterElementsUpdate;
}  // namespace ov::op::v12

namespace ov::op::v13 {
class BitwiseAnd;
class BitwiseNot;
class BitwiseOr;
class BitwiseXor;
class FakeConvert;
class Multinomial;
class NMSRotated;
class ScaledDotProductAttention;
}  // namespace ov::op::v13

namespace ov::op::v14 {
class AvgPool;
class ConvertPromoteTypes;
class Inverse;
class MaxPool;
}  // namespace ov::op::v14

namespace ov::op::v15 {
class BitwiseLeftShift;
class BitwiseRightShift;
class Col2Im;
class EmbeddingBagOffsets;
class EmbeddingBagPacked;
class ROIAlignRotated;
class STFT;
class ScatterNDUpdate;
class SearchSorted;
class SliceScatter;
class Squeeze;
class StringTensorPack;
class StringTensorUnpack;
}  // namespace ov::op::v15

namespace ov::op::v16 {
class ISTFT;
class Identity;
class SegmentMax;
class SparseFillEmptyRows;
}  // namespace ov::op::v16
