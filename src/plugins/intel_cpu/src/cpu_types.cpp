// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "cpu_types.h"

#include <vector>
#include <string>

namespace ov {
namespace intel_cpu {

using Dim = std::size_t;
using VectorDims = std::vector<Dim>;

const InferenceEngine::details::caseless_unordered_map<std::string, Type> type_to_name_tbl = {
        { "Constant", Type::Input },
        { "Parameter", Type::Input },
        { "Result", Type::Output },
        { "Eye", Type::Eye },
        { "Convolution", Type::Convolution },
        { "GroupConvolution", Type::Convolution },
        { "MatMul", Type::MatMul },
        { "FullyConnected", Type::FullyConnected },
        { "MaxPool", Type::Pooling },
        { "AvgPool", Type::Pooling },
        { "AdaptiveMaxPool", Type::AdaptivePooling},
        { "AdaptiveAvgPool", Type::AdaptivePooling},
        { "Add", Type::Eltwise },
        { "IsFinite", Type::Eltwise },
        { "IsInf", Type::Eltwise },
        { "IsNaN", Type::Eltwise },
        { "Subtract", Type::Eltwise },
        { "Multiply", Type::Eltwise },
        { "Divide", Type::Eltwise },
        { "SquaredDifference", Type::Eltwise },
        { "Maximum", Type::Eltwise },
        { "Minimum", Type::Eltwise },
        { "Mod", Type::Eltwise },
        { "FloorMod", Type::Eltwise },
        { "Power", Type::Eltwise },
        { "PowerStatic", Type::Eltwise },
        { "Equal", Type::Eltwise },
        { "NotEqual", Type::Eltwise },
        { "Greater", Type::Eltwise },
        { "GreaterEqual", Type::Eltwise },
        { "Less", Type::Eltwise },
        { "LessEqual", Type::Eltwise },
        { "LogicalAnd", Type::Eltwise },
        { "LogicalOr", Type::Eltwise },
        { "LogicalXor", Type::Eltwise },
        { "LogicalNot", Type::Eltwise },
        { "Relu", Type::Eltwise },
        { "LeakyRelu", Type::Eltwise },
        { "Gelu", Type::Eltwise },
        { "Elu", Type::Eltwise },
        { "Tanh", Type::Eltwise },
        { "Sigmoid", Type::Eltwise },
        { "Abs", Type::Eltwise },
        { "Sqrt", Type::Eltwise },
        { "Clamp", Type::Eltwise },
        { "Exp", Type::Eltwise },
        { "SwishCPU", Type::Eltwise },
        { "HSwish", Type::Eltwise },
        { "Mish", Type::Eltwise },
        { "HSigmoid", Type::Eltwise },
        { "Round", Type::Eltwise },
        { "PRelu", Type::Eltwise },
        { "Erf", Type::Eltwise },
        { "SoftPlus", Type::Eltwise },
        { "SoftSign", Type::Eltwise },
        { "Select", Type::Eltwise},
        { "Reshape", Type::Reshape },
        { "Squeeze", Type::Reshape },
        { "Unsqueeze", Type::Reshape },
        { "ShapeOf", Type::ShapeOf },
        { "NonZero", Type::NonZero },
        { "Softmax", Type::Softmax },
        { "Reorder", Type::Reorder },
        { "BatchToSpace", Type::BatchToSpace },
        { "SpaceToBatch", Type::SpaceToBatch },
        { "DepthToSpace", Type::DepthToSpace },
        { "SpaceToDepth", Type::SpaceToDepth },
        { "Roll", Type::Roll },
        { "LRN", Type::Lrn },
        { "Split", Type::Split },
        { "VariadicSplit", Type::Split },
        { "Concat", Type::Concatenation },
        { "ConvolutionBackpropData", Type::Deconvolution },
        { "GroupConvolutionBackpropData", Type::Deconvolution },
        { "StridedSlice", Type::StridedSlice },
        { "Slice", Type::StridedSlice },
        { "Tile", Type::Tile },
        { "ROIAlign", Type::ROIAlign },
        { "ROIPooling", Type::ROIPooling },
        { "PSROIPooling", Type::PSROIPooling },
        { "DeformablePSROIPooling", Type::PSROIPooling },
        { "Pad", Type::Pad },
        { "Transpose", Type::Transpose },
        { "LSTMCell", Type::RNNCell },
        { "GRUCell", Type::RNNCell },
        { "AUGRUCell", Type::RNNCell },
        { "RNNCell", Type::RNNCell },
        { "LSTMSequence", Type::RNNSeq },
        { "GRUSequence", Type::RNNSeq },
        { "AUGRUSequence", Type::RNNSeq },
        { "RNNSequence", Type::RNNSeq },
        { "FakeQuantize", Type::FakeQuantize },
        { "BinaryConvolution", Type::BinaryConvolution },
        { "DeformableConvolution", Type::DeformableConvolution },
        { "TensorIterator", Type::TensorIterator },
        { "Loop", Type::TensorIterator },
        { "ReadValue", Type::MemoryInput},  // for construction from name ctor, arbitrary name is used
        { "Assign", Type::MemoryOutput },  // for construction from layer ctor
        { "Convert", Type::Convert },
        { "NV12toRGB", Type::ColorConvert },
        { "NV12toBGR", Type::ColorConvert },
        { "I420toRGB", Type::ColorConvert },
        { "I420toBGR", Type::ColorConvert },
        { "MVN", Type::MVN},
        { "NormalizeL2", Type::NormalizeL2},
        { "ScatterUpdate", Type::ScatterUpdate},
        { "ScatterElementsUpdate", Type::ScatterElementsUpdate},
        { "ScatterNDUpdate", Type::ScatterNDUpdate},
        { "Interpolate", Type::Interpolate},
        { "ReduceL1", Type::Reduce},
        { "ReduceL2", Type::Reduce},
        { "ReduceLogicalAnd", Type::Reduce},
        { "ReduceLogicalOr", Type::Reduce},
        { "ReduceMax", Type::Reduce},
        { "ReduceMean", Type::Reduce},
        { "ReduceMin", Type::Reduce},
        { "ReduceProd", Type::Reduce},
        { "ReduceSum", Type::Reduce},
        { "ReduceLogSum", Type::Reduce},
        { "ReduceLogSumExp", Type::Reduce},
        { "ReduceSumSquare", Type::Reduce},
        { "Broadcast", Type::Broadcast},
        { "EmbeddingSegmentsSum", Type::EmbeddingSegmentsSum},
        { "EmbeddingBagPackedSum", Type::EmbeddingBagPackedSum},
        { "EmbeddingBagOffsetsSum", Type::EmbeddingBagOffsetsSum},
        { "Gather", Type::Gather},
        { "GatherElements", Type::GatherElements},
        { "GatherND", Type::GatherND},
        { "GridSample", Type::GridSample},
        { "OneHot", Type::OneHot},
        { "RegionYolo", Type::RegionYolo},
        { "ShuffleChannels", Type::ShuffleChannels},
        { "DFT", Type::DFT},
        { "IDFT", Type::DFT},
        { "RDFT", Type::RDFT},
        { "IRDFT", Type::RDFT},
        { "Abs", Type::Math},
        { "Acos", Type::Math},
        { "Acosh", Type::Math},
        { "Asin", Type::Math},
        { "Asinh", Type::Math},
        { "Atan", Type::Math},
        { "Atanh", Type::Math},
        { "Ceil", Type::Math},
        { "Ceiling", Type::Math},
        { "Cos", Type::Math},
        { "Cosh", Type::Math},
        { "Floor", Type::Math},
        { "HardSigmoid", Type::Math},
        { "If", Type::If},
        { "Log", Type::Math},
        { "Neg", Type::Math},
        { "Reciprocal", Type::Math},
        { "Selu", Type::Math},
        { "Sign", Type::Math},
        { "Sin", Type::Math},
        { "Sinh", Type::Math},
        { "SoftPlus", Type::Math},
        { "Softsign", Type::Math},
        { "Tan", Type::Math},
        { "CTCLoss", Type::CTCLoss},
        { "Bucketize", Type::Bucketize},
        { "CTCGreedyDecoder", Type::CTCGreedyDecoder},
        { "CTCGreedyDecoderSeqLen", Type::CTCGreedyDecoderSeqLen},
        { "CumSum", Type::CumSum},
        { "DetectionOutput", Type::DetectionOutput},
        { "ExperimentalDetectronDetectionOutput", Type::ExperimentalDetectronDetectionOutput},
        { "LogSoftmax", Type::LogSoftmax},
        { "TopK", Type::TopK},
        { "GatherTree", Type::GatherTree},
        { "GRN", Type::GRN},
        { "Range", Type::Range},
        { "Proposal", Type::Proposal},
        { "ReorgYolo", Type::ReorgYolo},
        { "ReverseSequence", Type::ReverseSequence},
        { "ExperimentalDetectronTopKROIs", Type::ExperimentalDetectronTopKROIs},
        { "ExperimentalDetectronROIFeatureExtractor", Type::ExperimentalDetectronROIFeatureExtractor},
        { "ExperimentalDetectronPriorGridGenerator", Type::ExperimentalDetectronPriorGridGenerator},
        { "ExperimentalDetectronGenerateProposalsSingleImage", Type::ExperimentalDetectronGenerateProposalsSingleImage},
        { "GenerateProposals", Type::GenerateProposals},
        { "ExtractImagePatches", Type::ExtractImagePatches},
        { "NonMaxSuppression", Type::NonMaxSuppression},
        { "NonMaxSuppressionIEInternal", Type::NonMaxSuppression},
        { "MatrixNms", Type::MatrixNms},
        { "MulticlassNms", Type::MulticlassNms},
        { "MulticlassNmsIEInternal", Type::MulticlassNms},
        { "Reference", Type::Reference},
        { "Subgraph", Type::Subgraph},
        { "PriorBox", Type::PriorBox},
        { "PriorBoxClustered", Type::PriorBoxClustered},
        {"Interaction", Type::Interaction},
        { "MHA", Type::MHA},
        { "Unique", Type::Unique}
};

Type TypeFromName(const std::string& type) {
    auto itType = type_to_name_tbl.find(type);
    if (type_to_name_tbl.end() != itType) {
        return itType->second;
    } else {
        return Type::Unknown;
    }
}

std::string NameFromType(const Type type) {
    switch (type) {
        case Type::Generic:
            return "Generic";
        case Type::Reorder:
            return "Reorder";
        case Type::Input:
            return "Input";
        case Type::Output:
            return "Output";
        case Type::Eye:
            return "Eye";
        case Type::Convolution:
            return "Convolution";
        case Type::Deconvolution:
            return "Deconvolution";
        case Type::Lrn:
            return "Lrn";
        case Type::Pooling:
            return "Pooling";
        case Type::AdaptivePooling:
            return "AdaptivePooling";
        case Type::FullyConnected:
            return "FullyConnected";
        case Type::MatMul:
            return "MatMul";
        case Type::Softmax:
            return "Softmax";
        case Type::Split:
            return "Split";
        case Type::Concatenation:
            return "Concatenation";
        case Type::StridedSlice:
            return "StridedSlice";
        case Type::Reshape:
            return "Reshape";
        case Type::ShapeOf:
            return "ShapeOf";
        case Type::NonZero:
            return "NonZero";
        case Type::Tile:
            return "Tile";
        case Type::ROIAlign:
            return "ROIAlign";
        case Type::ROIPooling:
            return "ROIPooling";
        case Type::PSROIPooling:
            return "PSROIPooling";
        case Type::DepthToSpace:
            return "DepthToSpace";
        case Type::BatchToSpace:
            return "BatchToSpace";
        case Type::Pad:
            return "Pad";
        case Type::Transpose:
            return "Transpose";
        case Type::SpaceToDepth:
            return "SpaceToDepth";
        case Type::SpaceToBatch:
            return "SpaceToBatch";
        case Type::MemoryOutput:
            return "MemoryOutput";
        case Type::MemoryInput:
            return "MemoryInput";
        case Type::RNNSeq:
            return "RNNSeq";
        case Type::RNNCell:
            return "RNNCell";
        case Type::Eltwise:
            return "Eltwise";
        case Type::FakeQuantize:
            return "FakeQuantize";
        case Type::BinaryConvolution:
            return "BinaryConvolution";
        case Type::DeformableConvolution:
            return "DeformableConvolution";
        case Type::MVN:
            return "MVN";
        case Type::TensorIterator:
            return "TensorIterator";
        case Type::Convert:
            return "Convert";
        case Type::ColorConvert:
            return "ColorConvert";
        case Type::NormalizeL2:
            return "NormalizeL2";
        case Type::ScatterUpdate:
            return "ScatterUpdate";
        case Type::ScatterElementsUpdate:
            return "ScatterElementsUpdate";
        case Type::ScatterNDUpdate:
            return "ScatterNDUpdate";
        case Type::Interaction:
            return "Interaction";
        case Type::Interpolate:
            return "Interpolate";
        case Type::Reduce:
            return "Reduce";
        case Type::Broadcast:
            return "Broadcast";
        case Type::EmbeddingSegmentsSum:
            return "EmbeddingSegmentsSum";
        case Type::EmbeddingBagPackedSum:
            return "EmbeddingBagPackedSum";
        case Type::EmbeddingBagOffsetsSum:
            return "EmbeddingBagOffsetsSum";
        case Type::Gather:
            return "Gather";
        case Type::GatherElements:
            return "GatherElements";
        case Type::GatherND:
            return "GatherND";
        case Type::GridSample:
            return "GridSample";
        case Type::OneHot:
            return "OneHot";
        case Type::RegionYolo:
            return "RegionYolo";
        case Type::Roll:
            return "Roll";
        case Type::ShuffleChannels:
            return "ShuffleChannels";
        case Type::DFT:
            return "DFT";
        case Type::RDFT:
            return "RDFT";
        case Type::Math:
            return "Math";
        case Type::CTCLoss:
            return "CTCLoss";
        case Type::Bucketize:
            return "Bucketize";
        case Type::CTCGreedyDecoder:
            return "CTCGreedyDecoder";
        case Type::CTCGreedyDecoderSeqLen:
            return "CTCGreedyDecoderSeqLen";
        case Type::CumSum:
            return "CumSum";
        case Type::DetectionOutput:
            return "DetectionOutput";
        case Type::ExperimentalDetectronDetectionOutput:
            return "ExperimentalDetectronDetectionOutput";
        case Type::If:
            return "If";
        case Type::LogSoftmax:
            return "LogSoftmax";
        case Type::TopK:
            return "TopK";
        case Type::GatherTree:
            return "GatherTree";
        case Type::GRN:
            return "GRN";
        case Type::Range:
            return "Range";
        case Type::Proposal:
            return "Proposal";
        case Type::ReorgYolo:
            return "ReorgYolo";
        case Type::ReverseSequence:
            return "ReverseSequence";
        case Type::ExperimentalDetectronTopKROIs:
            return "ExperimentalDetectronTopKROIs";
        case Type::ExperimentalDetectronROIFeatureExtractor:
            return "ExperimentalDetectronROIFeatureExtractor";
        case Type::ExperimentalDetectronPriorGridGenerator:
            return "ExperimentalDetectronPriorGridGenerator";
        case Type::ExperimentalDetectronGenerateProposalsSingleImage:
            return "ExperimentalDetectronGenerateProposalsSingleImage";
        case Type::GenerateProposals:
            return "GenerateProposals";
        case Type::ExtractImagePatches:
            return "ExtractImagePatches";
        case Type::NonMaxSuppression:
            return "NonMaxSuppression";
        case Type::MatrixNms:
            return "MatrixNms";
        case Type::MulticlassNms:
            return "MulticlassNms";
        case Type::Reference:
            return "Reference";
        case Type::Subgraph:
            return "Subgraph";
        case Type::MHA:
            return "MHA";
        case Type::Unique:
            return "Unique";
        default:
            return "Unknown";
    }
}

std::string algToString(const Algorithm alg) {
#define CASE(_alg) do {                         \
    if (alg == Algorithm::_alg) return #_alg;   \
} while (0)
    CASE(Default);
    CASE(PoolingMax);
    CASE(PoolingAvg);
    CASE(ConvolutionCommon);
    CASE(ConvolutionGrouped);
    CASE(DeconvolutionCommon);
    CASE(DeconvolutionGrouped);
    CASE(EltwiseAdd);
    CASE(EltwiseIsFinite);
    CASE(EltwiseIsInf);
    CASE(EltwiseIsNaN);
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
    CASE(EltwiseSelect);
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
    CASE(FQRequantization);
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

}   // namespace intel_cpu
}   // namespace ov

