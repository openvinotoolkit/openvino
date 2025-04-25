// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "cpu_types.h"

#include <sstream>
#include <string>

#include "cpu_shape.h"

namespace ov::intel_cpu {

std::string dim2str(Dim dim) {
    return dim == Shape::UNDEFINED_DIM ? "?" : std::to_string(dim);
}

std::string dims2str(const VectorDims& dims) {
    std::stringstream output;
    output << "{";

    if (!dims.empty()) {
        auto itr = dims.begin();
        do {
            output << dim2str(*itr);
        } while (++itr != dims.end() && output << ", ");
    }

    output << "}";
    return output.str();
}

using TypeToNameMap = ov::intel_cpu::caseless_unordered_map<std::string, Type>;

static const TypeToNameMap& get_type_to_name_tbl() {
    static const TypeToNameMap type_to_name_tbl = {
        {"Constant", Type::Input},
        {"Parameter", Type::Input},
        {"Result", Type::Output},
        {"Eye", Type::Eye},
        {"Convolution", Type::Convolution},
        {"GroupConvolution", Type::Convolution},
        {"MatMul", Type::MatMul},
        {"FullyConnected", Type::FullyConnected},
        {"FullyConnectedCompressed", Type::FullyConnected},
        {"FullyConnectedQuantizedLegacy", Type::FullyConnected},
        {"FullyConnectedQuantized", Type::FullyConnected},
        {"MaxPool", Type::Pooling},
        {"AvgPool", Type::Pooling},
        {"AdaptiveMaxPool", Type::AdaptivePooling},
        {"AdaptiveAvgPool", Type::AdaptivePooling},
        {"Add", Type::Eltwise},
        {"IsFinite", Type::Eltwise},
        {"IsInf", Type::Eltwise},
        {"IsNaN", Type::Eltwise},
        {"Subtract", Type::Eltwise},
        {"Multiply", Type::Eltwise},
        {"Divide", Type::Eltwise},
        {"SquaredDifference", Type::Eltwise},
        {"Maximum", Type::Eltwise},
        {"Minimum", Type::Eltwise},
        {"Mod", Type::Eltwise},
        {"FloorMod", Type::Eltwise},
        {"Power", Type::Eltwise},
        {"PowerStatic", Type::Eltwise},
        {"Equal", Type::Eltwise},
        {"NotEqual", Type::Eltwise},
        {"Greater", Type::Eltwise},
        {"GreaterEqual", Type::Eltwise},
        {"Less", Type::Eltwise},
        {"LessEqual", Type::Eltwise},
        {"LogicalAnd", Type::Eltwise},
        {"LogicalOr", Type::Eltwise},
        {"LogicalXor", Type::Eltwise},
        {"LogicalNot", Type::Eltwise},
        {"Relu", Type::Eltwise},
        {"LeakyRelu", Type::Eltwise},
        {"Gelu", Type::Eltwise},
        {"Elu", Type::Eltwise},
        {"Tanh", Type::Eltwise},
        {"Sigmoid", Type::Eltwise},
        {"Abs", Type::Eltwise},
        {"Sqrt", Type::Eltwise},
        {"Clamp", Type::Eltwise},
        {"Exp", Type::Eltwise},
        {"SwishCPU", Type::Eltwise},
        {"HSwish", Type::Eltwise},
        {"Mish", Type::Eltwise},
        {"HSigmoid", Type::Eltwise},
        {"Round", Type::Eltwise},
        {"PRelu", Type::Eltwise},
        {"Erf", Type::Eltwise},
        {"SoftPlus", Type::Eltwise},
        {"SoftSign", Type::Eltwise},
        {"SegmentMax", Type::SegmentMax},
        {"Select", Type::Eltwise},
        {"Log", Type::Eltwise},
        {"BitwiseAnd", Type::Eltwise},
        {"BitwiseNot", Type::Eltwise},
        {"BitwiseOr", Type::Eltwise},
        {"BitwiseXor", Type::Eltwise},
        {"BitwiseLeftShift", Type::Eltwise},
        {"BitwiseRightShift", Type::Eltwise},
        {"Reshape", Type::Reshape},
        {"Squeeze", Type::Reshape},
        {"Unsqueeze", Type::Reshape},
        {"ShapeOf", Type::ShapeOf},
        {"NonZero", Type::NonZero},
        {"Softmax", Type::Softmax},
        {"Reorder", Type::Reorder},
        {"BatchToSpace", Type::BatchToSpace},
        {"SpaceToBatch", Type::SpaceToBatch},
        {"DepthToSpace", Type::DepthToSpace},
        {"SpaceToDepth", Type::SpaceToDepth},
        {"Roll", Type::Roll},
        {"LRN", Type::Lrn},
        {"Split", Type::Split},
        {"VariadicSplit", Type::Split},
        {"Concat", Type::Concatenation},
        {"ConvolutionBackpropData", Type::Deconvolution},
        {"GroupConvolutionBackpropData", Type::Deconvolution},
        {"StridedSlice", Type::StridedSlice},
        {"Slice", Type::StridedSlice},
        {"SliceScatter", Type::StridedSlice},
        {"Tile", Type::Tile},
        {"ROIAlign", Type::ROIAlign},
        {"ROIAlignRotated", Type::ROIAlignRotated},
        {"ROIPooling", Type::ROIPooling},
        {"PSROIPooling", Type::PSROIPooling},
        {"DeformablePSROIPooling", Type::PSROIPooling},
        {"Pad", Type::Pad},
        {"Transpose", Type::Transpose},
        {"LSTMCell", Type::RNNCell},
        {"GRUCell", Type::RNNCell},
        {"AUGRUCell", Type::RNNCell},
        {"RNNCell", Type::RNNCell},
        {"LSTMSequence", Type::RNNSeq},
        {"GRUSequence", Type::RNNSeq},
        {"AUGRUSequence", Type::RNNSeq},
        {"RNNSequence", Type::RNNSeq},
        {"FakeQuantize", Type::FakeQuantize},
        {"BinaryConvolution", Type::BinaryConvolution},
        {"DeformableConvolution", Type::DeformableConvolution},
        {"TensorIterator", Type::TensorIterator},
        {"Loop", Type::TensorIterator},
        {"ReadValue", Type::MemoryInput},  // for construction from name ctor, arbitrary name is used
        {"Assign", Type::MemoryOutput},    // for construction from layer ctor
        {"ReadValueWithSubgraph", Type::MemoryInput},
        {"Convert", Type::Convert},
        {"NV12toRGB", Type::ColorConvert},
        {"NV12toBGR", Type::ColorConvert},
        {"I420toRGB", Type::ColorConvert},
        {"I420toBGR", Type::ColorConvert},
        {"Col2Im", Type::Col2Im},
        {"MVN", Type::MVN},
        {"NormalizeL2", Type::NormalizeL2},
        {"ScatterUpdate", Type::ScatterUpdate},
        {"ScatterElementsUpdate", Type::ScatterElementsUpdate},
        {"ScatterNDUpdate", Type::ScatterNDUpdate},
        {"StringTensorPack", Type::StringTensorPack},
        {"StringTensorUnpack", Type::StringTensorUnpack},
        {"Interpolate", Type::Interpolate},
        {"RandomUniform", Type::RandomUniform},
        {"ReduceL1", Type::Reduce},
        {"ReduceL2", Type::Reduce},
        {"ReduceLogicalAnd", Type::Reduce},
        {"ReduceLogicalOr", Type::Reduce},
        {"ReduceMax", Type::Reduce},
        {"ReduceMean", Type::Reduce},
        {"ReduceMin", Type::Reduce},
        {"ReduceProd", Type::Reduce},
        {"ReduceSum", Type::Reduce},
        {"ReduceLogSum", Type::Reduce},
        {"ReduceLogSumExp", Type::Reduce},
        {"ReduceSumSquare", Type::Reduce},
        {"Broadcast", Type::Broadcast},
        {"EmbeddingSegmentsSum", Type::EmbeddingSegmentsSum},
        {"EmbeddingBagPackedSum", Type::EmbeddingBagPacked},
        {"EmbeddingBagOffsetsSum", Type::EmbeddingBagOffsets},
        {"Gather", Type::Gather},
        {"GatherElements", Type::GatherElements},
        {"GatherND", Type::GatherND},
        {"GridSample", Type::GridSample},
        {"OneHot", Type::OneHot},
        {"RegionYolo", Type::RegionYolo},
        {"ShuffleChannels", Type::ShuffleChannels},
        {"DFT", Type::DFT},
        {"IDFT", Type::DFT},
        {"RDFT", Type::RDFT},
        {"IRDFT", Type::RDFT},
        {"ISTFT", Type::ISTFT},
        {"STFT", Type::STFT},
        {"Abs", Type::Math},
        {"Acos", Type::Math},
        {"Acosh", Type::Math},
        {"Asin", Type::Math},
        {"Asinh", Type::Math},
        {"Atan", Type::Math},
        {"Atanh", Type::Math},
        {"Ceil", Type::Math},
        {"Ceiling", Type::Eltwise},
        {"Negative", Type::Eltwise},
        {"Cos", Type::Math},
        {"Cosh", Type::Math},
        {"Floor", Type::Eltwise},
        {"HardSigmoid", Type::Math},
        {"If", Type::If},
        {"Reciprocal", Type::Math},
        {"Selu", Type::Math},
        {"Sign", Type::Math},
        {"Sin", Type::Math},
        {"Sinh", Type::Math},
        {"SoftPlus", Type::Math},
        {"Softsign", Type::Math},
        {"Tan", Type::Math},
        {"CTCLoss", Type::CTCLoss},
        {"Bucketize", Type::Bucketize},
        {"CTCGreedyDecoder", Type::CTCGreedyDecoder},
        {"CTCGreedyDecoderSeqLen", Type::CTCGreedyDecoderSeqLen},
        {"CumSum", Type::CumSum},
        {"DetectionOutput", Type::DetectionOutput},
        {"ExperimentalDetectronDetectionOutput", Type::ExperimentalDetectronDetectionOutput},
        {"LogSoftmax", Type::LogSoftmax},
        {"TopK", Type::TopK},
        {"GatherTree", Type::GatherTree},
        {"GRN", Type::GRN},
        {"Range", Type::Range},
        {"Proposal", Type::Proposal},
        {"ReorgYolo", Type::ReorgYolo},
        {"ReverseSequence", Type::ReverseSequence},
        {"ExperimentalDetectronTopKROIs", Type::ExperimentalDetectronTopKROIs},
        {"ExperimentalDetectronROIFeatureExtractor", Type::ExperimentalDetectronROIFeatureExtractor},
        {"ExperimentalDetectronPriorGridGenerator", Type::ExperimentalDetectronPriorGridGenerator},
        {"ExperimentalDetectronGenerateProposalsSingleImage", Type::ExperimentalDetectronGenerateProposalsSingleImage},
        {"ExtractImagePatches", Type::ExtractImagePatches},
        {"GenerateProposals", Type::GenerateProposals},
        {"Inverse", Type::Inverse},
        {"NonMaxSuppression", Type::NonMaxSuppression},
        {"NonMaxSuppressionIEInternal", Type::NonMaxSuppression},
        {"NMSRotated", Type::NonMaxSuppression},
        {"MatrixNms", Type::MatrixNms},
        {"MulticlassNms", Type::MulticlassNms},
        {"MulticlassNmsIEInternal", Type::MulticlassNms},
        {"Multinomial", Type::Multinomial},
        {"Reference", Type::Reference},
        {"Subgraph", Type::Subgraph},
        {"SubModel", Type::SubModel},
        {"PriorBox", Type::PriorBox},
        {"PriorBoxClustered", Type::PriorBoxClustered},
        {"Interaction", Type::Interaction},
        {"MHA", Type::MHA},
        {"Unique", Type::Unique},
        {"Ngram", Type::Ngram},
        {"ScaledDotProductAttention", Type::ScaledDotProductAttention},
        {"ScaledDotProductAttentionWithKVCache", Type::ScaledDotProductAttention},
        {"SDPAWithTransposeReshape", Type::ScaledDotProductAttention},
        {"PagedAttentionExtension", Type::PagedAttention},
        {"RoPE", Type::RoPE},
        {"GatherCompressed", Type::Gather},
        {"CausalMaskPreprocess", Type::CausalMaskPreprocess},
        {"EmbeddingBagPacked", Type::EmbeddingBagPacked},
        {"EmbeddingBagOffsets", Type::EmbeddingBagOffsets},
        {"LLMMLP", Type::LLMMLP},
        {"QKVProjection", Type::QKVProjection},
        {"RMS", Type::RMS},
        {"SearchSorted", Type::SearchSorted},
        {"LoraSubgraph", Type::LoRA}};
    return type_to_name_tbl;
}

Type TypeFromName(const std::string& type) {
    const TypeToNameMap& type_to_name_tbl = get_type_to_name_tbl();
    auto itType = type_to_name_tbl.find(type);
    if (type_to_name_tbl.end() != itType) {
        return itType->second;
    }
    return Type::Unknown;
}

std::string NameFromType(const Type type) {
#define CASE(_alg)   \
    case Type::_alg: \
        return #_alg;
    switch (type) {
        CASE(Reorder);
        CASE(Input);
        CASE(Output);
        CASE(Eye);
        CASE(Convolution);
        CASE(Deconvolution);
        CASE(Lrn);
        CASE(Pooling);
        CASE(AdaptivePooling);
        CASE(FullyConnected);
        CASE(MatMul);
        CASE(Softmax);
        CASE(Split);
        CASE(Concatenation);
        CASE(StridedSlice);
        CASE(Reshape);
        CASE(ShapeOf);
        CASE(NonZero);
        CASE(Tile);
        CASE(ROIAlign);
        CASE(ROIAlignRotated);
        CASE(ROIPooling);
        CASE(PSROIPooling);
        CASE(DepthToSpace);
        CASE(BatchToSpace);
        CASE(Pad);
        CASE(Transpose);
        CASE(SpaceToDepth);
        CASE(SpaceToBatch);
        CASE(MemoryOutput);
        CASE(MemoryInput);
        CASE(RNNSeq);
        CASE(RNNCell);
        CASE(Eltwise);
        CASE(FakeQuantize);
        CASE(BinaryConvolution);
        CASE(DeformableConvolution);
        CASE(MVN);
        CASE(TensorIterator);
        CASE(Convert);
        CASE(Col2Im);
        CASE(ColorConvert);
        CASE(NormalizeL2);
        CASE(ScatterUpdate);
        CASE(ScatterElementsUpdate);
        CASE(ScatterNDUpdate);
        CASE(StringTensorPack);
        CASE(StringTensorUnpack);
        CASE(Interaction);
        CASE(Interpolate);
        CASE(Reduce);
        CASE(Broadcast);
        CASE(EmbeddingSegmentsSum);
        CASE(EmbeddingBagPacked);
        CASE(EmbeddingBagOffsets);
        CASE(EmbeddingBagPackedSum);
        CASE(EmbeddingBagOffsetsSum);
        CASE(Gather);
        CASE(GatherElements);
        CASE(GatherND);
        CASE(GridSample);
        CASE(OneHot);
        CASE(RegionYolo);
        CASE(Roll);
        CASE(ShuffleChannels);
        CASE(DFT);
        CASE(RDFT);
        CASE(STFT);
        CASE(ISTFT);
        CASE(Math);
        CASE(CTCLoss);
        CASE(Bucketize);
        CASE(CTCGreedyDecoder);
        CASE(CTCGreedyDecoderSeqLen);
        CASE(CumSum);
        CASE(DetectionOutput);
        CASE(ExperimentalDetectronDetectionOutput);
        CASE(If);
        CASE(LogSoftmax);
        CASE(TopK);
        CASE(GatherTree);
        CASE(GRN);
        CASE(Range);
        CASE(Proposal);
        CASE(ReorgYolo);
        CASE(ReverseSequence);
        CASE(ExperimentalDetectronTopKROIs);
        CASE(ExperimentalDetectronROIFeatureExtractor);
        CASE(ExperimentalDetectronPriorGridGenerator);
        CASE(ExperimentalDetectronGenerateProposalsSingleImage);
        CASE(GenerateProposals);
        CASE(Inverse);
        CASE(ExtractImagePatches);
        CASE(NonMaxSuppression);
        CASE(MatrixNms);
        CASE(MulticlassNms);
        CASE(Multinomial);
        CASE(Reference);
        CASE(Subgraph);
        CASE(SubModel);
        CASE(PriorBox);
        CASE(PriorBoxClustered)
        CASE(MHA);
        CASE(RandomUniform);
        CASE(Unique);
        CASE(Ngram);
        CASE(ScaledDotProductAttention);
        CASE(PagedAttention);
        CASE(RoPE);
        CASE(CausalMaskPreprocess);
        CASE(LLMMLP);
        CASE(QKVProjection);
        CASE(RMS);
        CASE(SearchSorted);
        CASE(SegmentMax);
        CASE(LoRA);
        CASE(Unknown);
    }
#undef CASE
    return "Unknown";
}

std::string algToString(const Algorithm alg) {
#define CASE(_alg)        \
    case Algorithm::_alg: \
        return #_alg;
    switch (alg) {
        CASE(Default);
        CASE(PoolingMax);
        CASE(PoolingAvg);
        CASE(AdaptivePoolingMax);
        CASE(AdaptivePoolingAvg);
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
        CASE(EltwiseFloor);
        CASE(EltwiseCeiling);
        CASE(EltwiseFloorMod);
        CASE(EltwiseNegative);
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
        CASE(EltwiseGeluErf);
        CASE(EltwiseGeluTanh);
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
        CASE(EltwiseSoftSign);
        CASE(EltwiseLog);
        CASE(EltwiseBitwiseAnd);
        CASE(EltwiseBitwiseNot);
        CASE(EltwiseBitwiseOr);
        CASE(EltwiseBitwiseXor);
        CASE(EltwiseBitwiseLeftShift);
        CASE(EltwiseBitwiseRightShift);
        CASE(FQCommon);
        CASE(FQQuantization);
        CASE(FQBinarization);
        CASE(FullyConnectedCommon);
        CASE(FullyConnectedCompressed);
        CASE(FullyConnectedQuantized);
        CASE(FullyConnectedQuantizedLegacy);
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
    }
#undef CASE
    return "Undefined";
}

}  // namespace ov::intel_cpu
