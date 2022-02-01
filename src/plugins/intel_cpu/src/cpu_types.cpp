// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "cpu_types.h"

#include <vector>
#include <string>

namespace MKLDNNPlugin {

using Dim = std::size_t;
using VectorDims = std::vector<Dim>;

const InferenceEngine::details::caseless_unordered_map<std::string, Type> type_to_name_tbl = {
        { "Constant", Input },
        { "Parameter", Input },
        { "Result", Output },
        { "Convolution", Convolution },
        { "GroupConvolution", Convolution },
        { "MatMul", MatMul },
        { "FullyConnected", FullyConnected },
        { "MaxPool", Pooling },
        { "AvgPool", Pooling },
        { "AdaptiveMaxPool", AdaptivePooling},
        { "AdaptiveAvgPool", AdaptivePooling},
        { "Add", Eltwise },
        { "Subtract", Eltwise },
        { "Multiply", Eltwise },
        { "Divide", Eltwise },
        { "SquaredDifference", Eltwise },
        { "Maximum", Eltwise },
        { "Minimum", Eltwise },
        { "Mod", Eltwise },
        { "FloorMod", Eltwise },
        { "Power", Eltwise },
        { "PowerStatic", Eltwise },
        { "Equal", Eltwise },
        { "NotEqual", Eltwise },
        { "Greater", Eltwise },
        { "GreaterEqual", Eltwise },
        { "Less", Eltwise },
        { "LessEqual", Eltwise },
        { "LogicalAnd", Eltwise },
        { "LogicalOr", Eltwise },
        { "LogicalXor", Eltwise },
        { "LogicalNot", Eltwise },
        { "Relu", Eltwise },
        { "LeakyRelu", Eltwise },
        { "Gelu", Eltwise },
        { "Elu", Eltwise },
        { "Tanh", Eltwise },
        { "Sigmoid", Eltwise },
        { "Abs", Eltwise },
        { "Sqrt", Eltwise },
        { "Clamp", Eltwise },
        { "Exp", Eltwise },
        { "SwishCPU", Eltwise },
        { "HSwish", Eltwise },
        { "Mish", Eltwise },
        { "HSigmoid", Eltwise },
        { "Round", Eltwise },
        { "PRelu", Eltwise },
        { "Erf", Eltwise },
        { "SoftPlus", Eltwise },
        { "Reshape", Reshape },
        { "Squeeze", Reshape },
        { "Unsqueeze", Reshape },
        { "ShapeOf", ShapeOf },
        { "NonZero", NonZero },
        { "Softmax", Softmax },
        { "Reorder", Reorder },
        { "BatchToSpace", BatchToSpace },
        { "SpaceToBatch", SpaceToBatch },
        { "DepthToSpace", DepthToSpace },
        { "SpaceToDepth", SpaceToDepth },
        { "Roll", Roll },
        { "LRN", Lrn },
        { "Split", Split },
        { "VariadicSplit", Split },
        { "Concat", Concatenation },
        { "ConvolutionBackpropData", Deconvolution },
        { "GroupConvolutionBackpropData", Deconvolution },
        { "StridedSlice", StridedSlice },
        { "Slice", StridedSlice },
        { "Tile", Tile },
        { "ROIAlign", ROIAlign },
        { "ROIPooling", ROIPooling },
        { "PSROIPooling", PSROIPooling },
        { "DeformablePSROIPooling", PSROIPooling },
        { "Pad", Pad },
        { "Transpose", Transpose },
        { "LSTMCell", RNNCell },
        { "GRUCell", RNNCell },
        { "RNNCell", RNNCell },
        { "LSTMSequence", RNNSeq },
        { "GRUSequence", RNNSeq },
        { "RNNSequence", RNNSeq },
        { "FakeQuantize", FakeQuantize },
        { "BinaryConvolution", BinaryConvolution },
        { "DeformableConvolution", DeformableConvolution },
        { "TensorIterator", TensorIterator },
        { "Loop", TensorIterator },
        { "ReadValue", MemoryInput},  // for construction from name ctor, arbitrary name is used
        { "Assign", MemoryOutput },  // for construction from layer ctor
        { "Convert", Convert },
        { "NV12toRGB", ColorConvert },
        { "NV12toBGR", ColorConvert },
        { "I420toRGB", ColorConvert },
        { "I420toBGR", ColorConvert },
        { "MVN", MVN},
        { "NormalizeL2", NormalizeL2},
        { "ScatterUpdate", ScatterUpdate},
        { "ScatterElementsUpdate", ScatterElementsUpdate},
        { "ScatterNDUpdate", ScatterNDUpdate},
        { "Interpolate", Interpolate},
        { "ReduceL1", Reduce},
        { "ReduceL2", Reduce},
        { "ReduceLogicalAnd", Reduce},
        { "ReduceLogicalOr", Reduce},
        { "ReduceMax", Reduce},
        { "ReduceMean", Reduce},
        { "ReduceMin", Reduce},
        { "ReduceProd", Reduce},
        { "ReduceSum", Reduce},
        { "ReduceLogSum", Reduce},
        { "ReduceLogSumExp", Reduce},
        { "ReduceSumSquare", Reduce},
        { "Broadcast", Broadcast},
        { "EmbeddingSegmentsSum", EmbeddingSegmentsSum},
        { "EmbeddingBagPackedSum", EmbeddingBagPackedSum},
        { "EmbeddingBagOffsetsSum", EmbeddingBagOffsetsSum},
        { "Gather", Gather},
        { "GatherElements", GatherElements},
        { "GatherND", GatherND},
        { "OneHot", OneHot},
        { "RegionYolo", RegionYolo},
        { "Select", Select},
        { "ShuffleChannels", ShuffleChannels},
        { "DFT", DFT},
        { "IDFT", DFT},
        { "Abs", Math},
        { "Acos", Math},
        { "Acosh", Math},
        { "Asin", Math},
        { "Asinh", Math},
        { "Atan", Math},
        { "Atanh", Math},
        { "Ceil", Math},
        { "Ceiling", Math},
        { "Cos", Math},
        { "Cosh", Math},
        { "Floor", Math},
        { "HardSigmoid", Math},
        { "If", If},
        { "Log", Math},
        { "Neg", Math},
        { "Reciprocal", Math},
        { "Selu", Math},
        { "Sign", Math},
        { "Sin", Math},
        { "Sinh", Math},
        { "SoftPlus", Math},
        { "Softsign", Math},
        { "Tan", Math},
        { "CTCLoss", CTCLoss},
        { "Bucketize", Bucketize},
        { "CTCGreedyDecoder", CTCGreedyDecoder},
        { "CTCGreedyDecoderSeqLen", CTCGreedyDecoderSeqLen},
        { "CumSum", CumSum},
        { "DetectionOutput", DetectionOutput},
        { "ExperimentalDetectronDetectionOutput", ExperimentalDetectronDetectionOutput},
        { "LogSoftmax", LogSoftmax},
        { "TopK", TopK},
        { "GatherTree", GatherTree},
        { "GRN", GRN},
        { "Range", Range},
        { "Proposal", Proposal},
        { "ReorgYolo", ReorgYolo},
        { "ReverseSequence", ReverseSequence},
        { "ExperimentalDetectronTopKROIs", ExperimentalDetectronTopKROIs},
        { "ExperimentalDetectronROIFeatureExtractor", ExperimentalDetectronROIFeatureExtractor},
        { "ExperimentalDetectronPriorGridGenerator", ExperimentalDetectronPriorGridGenerator},
        { "ExperimentalDetectronGenerateProposalsSingleImage", ExperimentalDetectronGenerateProposalsSingleImage},
        { "ExtractImagePatches", ExtractImagePatches},
        { "NonMaxSuppression", NonMaxSuppression},
        { "NonMaxSuppressionIEInternal", NonMaxSuppression},
        { "MatrixNms", MatrixNms},
        { "MulticlassNms", MulticlassNms},
        { "Reference", Reference},
        { "Subgraph", Subgraph},
        { "PriorBox", PriorBox},
        { "PriorBoxClustered", PriorBoxClustered},
};

Type TypeFromName(const std::string& type) {
    auto itType = type_to_name_tbl.find(type);
    if (type_to_name_tbl.end() != itType) {
        return itType->second;
    } else {
        return Unknown;
    }
}

std::string NameFromType(const Type type) {
    switch (type) {
        case Generic:
            return "Generic";
        case Reorder:
            return "Reorder";
        case Input:
            return "Input";
        case Output:
            return "Output";
        case Convolution:
            return "Convolution";
        case Deconvolution:
            return "Deconvolution";
        case Lrn:
            return "Lrn";
        case Pooling:
            return "Pooling";
        case AdaptivePooling:
            return "AdaptivePooling";
        case FullyConnected:
            return "FullyConnected";
        case MatMul:
            return "MatMul";
        case Softmax:
            return "Softmax";
        case Split:
            return "Split";
        case Concatenation:
            return "Concatenation";
        case StridedSlice:
            return "StridedSlice";
        case Reshape:
            return "Reshape";
        case ShapeOf:
            return "ShapeOf";
        case NonZero:
            return "NonZero";
        case Tile:
            return "Tile";
        case ROIAlign:
            return "ROIAlign";
        case ROIPooling:
            return "ROIPooling";
        case PSROIPooling:
            return "PSROIPooling";
        case DepthToSpace:
            return "DepthToSpace";
        case BatchToSpace:
            return "BatchToSpace";
        case Pad:
            return "Pad";
        case Transpose:
            return "Transpose";
        case SpaceToDepth:
            return "SpaceToDepth";
        case SpaceToBatch:
            return "SpaceToBatch";
        case MemoryOutput:
            return "MemoryOutput";
        case MemoryInput:
            return "MemoryInput";
        case RNNSeq:
            return "RNNSeq";
        case RNNCell:
            return "RNNCell";
        case Eltwise:
            return "Eltwise";
        case FakeQuantize:
            return "FakeQuantize";
        case BinaryConvolution:
            return "BinaryConvolution";
        case DeformableConvolution:
            return "DeformableConvolution";
        case MVN:
            return "MVN";
        case TensorIterator:
            return "TensorIterator";
        case Convert:
            return "Convert";
        case ColorConvert:
            return "ColorConvert";
        case NormalizeL2:
            return "NormalizeL2";
        case ScatterUpdate:
            return "ScatterUpdate";
        case ScatterElementsUpdate:
            return "ScatterElementsUpdate";
        case ScatterNDUpdate:
            return "ScatterNDUpdate";
        case Interpolate:
            return "Interpolate";
        case Reduce:
            return "Reduce";
        case Broadcast:
            return "Broadcast";
        case EmbeddingSegmentsSum:
            return "EmbeddingSegmentsSum";
        case EmbeddingBagPackedSum:
            return "EmbeddingBagPackedSum";
        case EmbeddingBagOffsetsSum:
            return "EmbeddingBagOffsetsSum";
        case Gather:
            return "Gather";
        case GatherElements:
            return "GatherElements";
        case GatherND:
            return "GatherND";
        case OneHot:
            return "OneHot";
        case RegionYolo:
            return "RegionYolo";
        case Select:
            return "Select";
        case Roll:
            return "Roll";
        case ShuffleChannels:
            return "ShuffleChannels";
        case DFT:
            return "DFT";
        case Math:
            return "Math";
        case CTCLoss:
            return "CTCLoss";
        case Bucketize:
            return "Bucketize";
        case CTCGreedyDecoder:
            return "CTCGreedyDecoder";
        case CTCGreedyDecoderSeqLen:
            return "CTCGreedyDecoderSeqLen";
        case CumSum:
            return "CumSum";
        case DetectionOutput:
            return "DetectionOutput";
        case ExperimentalDetectronDetectionOutput:
            return "ExperimentalDetectronDetectionOutput";
        case If:
            return "If";
        case LogSoftmax:
            return "LogSoftmax";
        case TopK:
            return "TopK";
        case GatherTree:
            return "GatherTree";
        case GRN:
            return "GRN";
        case Range:
            return "Range";
        case Proposal:
            return "Proposal";
        case ReorgYolo:
            return "ReorgYolo";
        case ReverseSequence:
            return "ReverseSequence";
        case ExperimentalDetectronTopKROIs:
            return "ExperimentalDetectronTopKROIs";
        case ExperimentalDetectronROIFeatureExtractor:
            return "ExperimentalDetectronROIFeatureExtractor";
        case ExperimentalDetectronPriorGridGenerator:
            return "ExperimentalDetectronPriorGridGenerator";
        case ExperimentalDetectronGenerateProposalsSingleImage:
            return "ExperimentalDetectronGenerateProposalsSingleImage";
        case ExtractImagePatches:
            return "ExtractImagePatches";
        case NonMaxSuppression:
            return "NonMaxSuppression";
        case MatrixNms:
            return "MatrixNms";
        case MulticlassNms:
            return "MulticlassNms";
        case Reference:
            return "Reference";
        case Subgraph:
            return "Subgraph";
        default:
            return "Unknown";
    }
}

std::string algToString(const Algorithm alg) {
#define CASE(_alg) do {                     \
    if (alg == _alg) return #_alg;          \
} while (0)
    CASE(Default);
    CASE(PoolingMax);
    CASE(PoolingAvg);
    CASE(ConvolutionCommon);
    CASE(ConvolutionGrouped);
    CASE(DeconvolutionCommon);
    CASE(DeconvolutionGrouped);
    CASE(EltwiseAdd);
    CASE(EltwiseMultiply);
    CASE(EltwiseSubtract);
    CASE(EltwiseDivide);
    CASE(EltwiseFloorMod);
    CASE(EltwiseMod);
    CASE(EltwiseMaximum);
    CASE(EltwiseMinimum);
    CASE(EltwiseSquaredDifference);
    CASE(EltwisePowerDynamic);
    CASE(EltwisePowerStatic);
    CASE(EltwiseMulAdd);
    CASE(EltwiseEqual);
    CASE(EltwiseNotEqual);
    CASE(EltwiseGreater);
    CASE(EltwiseGreaterEqual);
    CASE(EltwiseLess);
    CASE(EltwiseLessEqual);
    CASE(EltwiseLogicalAnd);
    CASE(EltwiseLogicalOr);
    CASE(EltwiseLogicalXor);
    CASE(EltwiseLogicalNot);
    CASE(EltwiseRelu);
    CASE(EltwiseGelu);
    CASE(EltwiseElu);
    CASE(EltwiseTanh);
    CASE(EltwiseSigmoid);
    CASE(EltwiseAbs);
    CASE(EltwiseSqrt);
    CASE(EltwiseSoftRelu);
    CASE(EltwiseExp);
    CASE(EltwiseClamp);
    CASE(EltwiseSwish);
    CASE(EltwisePrelu);
    CASE(EltwiseMish);
    CASE(EltwiseHswish);
    CASE(EltwiseHsigmoid);
    CASE(EltwiseRoundHalfToEven);
    CASE(EltwiseRoundHalfAwayFromZero);
    CASE(EltwiseErf);
    CASE(FQCommon);
    CASE(FQQuantization);
    CASE(FQBinarization);
    CASE(ROIPoolingMax);
    CASE(ROIPoolingBilinear);
    CASE(ROIAlignMax);
    CASE(ROIAlignAvg);
    CASE(PSROIPoolingAverage);
    CASE(PSROIPoolingBilinear);
    CASE(PSROIPoolingBilinearDeformable);
    CASE(ReduceL1);
    CASE(ReduceL2);
    CASE(ReduceAnd);
    CASE(ReduceOr);
    CASE(ReduceMax);
    CASE(ReduceMean);
    CASE(ReduceMin);
    CASE(ReduceProd);
    CASE(ReduceSum);
    CASE(ReduceLogSum);
    CASE(ReduceLogSumExp);
    CASE(ReduceSumSquare);
    CASE(MathAbs);
    CASE(MathAcos);
    CASE(MathAcosh);
    CASE(MathAsin);
    CASE(MathAsinh);
    CASE(MathAtan);
    CASE(MathAtanh);
    CASE(MathCeiling);
    CASE(MathCos);
    CASE(MathCosh);
    CASE(MathErf);
    CASE(MathFloor);
    CASE(MathHardSigmoid);
    CASE(MathLog);
    CASE(MathNegative);
    CASE(MathReciprocal);
    CASE(MathSelu);
    CASE(MathSign);
    CASE(MathSin);
    CASE(MathSinh);
    CASE(MathSoftPlus);
    CASE(MathSoftsign);
    CASE(MathTan);
    CASE(TensorIteratorCommon);
    CASE(TensorIteratorLoop);
    CASE(ColorConvertNV12toRGB);
    CASE(ColorConvertNV12toBGR);
    CASE(ColorConvertI420toRGB);
    CASE(ColorConvertI420toBGR);
#undef CASE
    return "Undefined";
}

} // namespace MKLDNNPlugin
