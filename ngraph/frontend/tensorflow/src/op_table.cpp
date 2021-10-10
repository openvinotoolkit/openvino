// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tf;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {
#define OP_CONVERTER(op) OutputVector op(const NodeContext& node)
#define OP_T_CONVERTER(op) \
    template <class T>     \
    OutputVector op(const NodeContext& node)

OP_T_CONVERTER(TranslateUnaryOp);
OP_T_CONVERTER(TranslateBinaryOp);
OP_T_CONVERTER(TranslateDirectReduceOp);

OP_CONVERTER(TranslateAddNOp);
OP_CONVERTER(TranslateArgMaxOp);
OP_CONVERTER(TranslateArgMinOp);
OP_CONVERTER(TranslateAvgPoolOp);
OP_CONVERTER(TranslateBiasAddOp);
OP_CONVERTER(TranslateBatchNDAndSpaceNDOp);
OP_CONVERTER(TranslateCastOp);
OP_CONVERTER(TranslateConcatOp);
OP_CONVERTER(TranslateConstOp);
OP_CONVERTER(TranslateConv2DOp);
OP_CONVERTER(TranslateConv2DBackpropInputOp);
OP_CONVERTER(TranslateConv3DOp);
OP_CONVERTER(TranslateCumsumOp);
OP_CONVERTER(TranslateCropAndResizeOp);
OP_CONVERTER(TranslateDepthToSpaceOp);
OP_CONVERTER(TranslateDepthwiseConv2dNativeOp);
OP_CONVERTER(TranslateEluOp);
OP_CONVERTER(TranslateExpandDimsOp);
OP_CONVERTER(TranslateFakeQuantWithMinMaxVarsOp);
OP_CONVERTER(TranslateFillOp);
OP_CONVERTER(TranslateFloorDivOp);
OP_CONVERTER(TranslateFusedBatchNormOp);
OP_CONVERTER(TranslateGatherOp);
OP_CONVERTER(TranslateGatherV2Op);
OP_CONVERTER(TranslateGatherNdOp);
OP_CONVERTER(TranslateFusedConv2DOp);
OP_CONVERTER(TranslateFusedMatMulOp);
OP_CONVERTER(TranslateIdentityOp);
OP_CONVERTER(TranslateInterpolateOp);
OP_CONVERTER(TranslateIsFiniteOp);
OP_CONVERTER(TranslateL2LossOp);
OP_CONVERTER(TranslateLogSoftmaxOp);
OP_CONVERTER(TranslateLog1pOp);
OP_CONVERTER(TranslateLRNOp);
OP_CONVERTER(TranslateMatMulOp);
OP_CONVERTER(TranslateMaxPoolOp);
OP_CONVERTER(TranslateNonMaxSuppressionOp);
OP_CONVERTER(TranslatePadOp);
OP_CONVERTER(PlaceholderOp);
OP_CONVERTER(NoOp);
OP_CONVERTER(TranslateOneHotOp);
OP_CONVERTER(TranslatePackOp);
OP_CONVERTER(TranslateRangeOp);
OP_CONVERTER(TranslateRankOp);
OP_CONVERTER(TranslateRandomUniformOp);
OP_CONVERTER(TranslateRelu6Op);
OP_CONVERTER(TranslateReciprocalOp);
OP_CONVERTER(TranslateReshapeOp);
OP_CONVERTER(RetvalOp);
OP_CONVERTER(TranslateReverseOp);
OP_CONVERTER(TranslateRollOp);
OP_CONVERTER(TranslateRoundOp);
OP_CONVERTER(TranslateRsqrtOp);
OP_CONVERTER(TranslateSelectOp);
OP_CONVERTER(TranslateShapeOp);
OP_CONVERTER(TranslateSizeOp);
OP_CONVERTER(TranslateSliceOp);
OP_CONVERTER(TranslateSoftmaxOp);
OP_CONVERTER(TranslateSpaceToDepthOp);
OP_CONVERTER(TranslateSplitOp);
OP_CONVERTER(TranslateSplitVOp);
OP_CONVERTER(TranslateSquareOp);
OP_CONVERTER(TranslateSqueezeOp);
OP_CONVERTER(TranslateStridedSliceOp);
OP_CONVERTER(TranslateSqrtOp);
OP_CONVERTER(TranslateTileOp);
OP_CONVERTER(TranslateTopKV2Op);
OP_CONVERTER(TranslateTransposeOp);
OP_CONVERTER(TranslateUnpackOp);
OP_CONVERTER(TranslateWhereOp);
OP_CONVERTER(TranslateXdivyOp);
OP_CONVERTER(TranslateZerosLikeOp);

const std::map<const std::string, const CreatorFunction> get_supported_ops() {
    return {
        // note: UnaryOp translator declaration for each op must to be added in unary_op.cpp file
        {"Abs", TranslateUnaryOp<opset8::Abs>},
        {"Acos", TranslateUnaryOp<opset8::Acos>},
        {"Acosh", TranslateUnaryOp<opset8::Acosh>},
        {"Asin", TranslateUnaryOp<opset8::Asin>},
        {"Asinh", TranslateUnaryOp<opset8::Asinh>},
        {"Atan", TranslateUnaryOp<opset8::Atan>},
        {"Atanh", TranslateUnaryOp<opset8::Atanh>},
        {"Ceil", TranslateUnaryOp<opset8::Ceiling>},
        {"Cos", TranslateUnaryOp<opset8::Cos>},
        {"Cosh", TranslateUnaryOp<opset8::Cosh>},
        {"Exp", TranslateUnaryOp<opset8::Exp>},
        {"Floor", TranslateUnaryOp<opset8::Floor>},
        {"Log", TranslateUnaryOp<opset8::Log>},
        {"LogicalNot", TranslateUnaryOp<opset8::LogicalNot>},
        {"Neg", TranslateUnaryOp<opset8::Negative>},
        {"Relu", TranslateUnaryOp<opset8::Relu>},
        {"Sigmoid", TranslateUnaryOp<opset8::Sigmoid>},
        {"Sin", TranslateUnaryOp<opset8::Sin>},
        {"Sinh", TranslateUnaryOp<opset8::Sinh>},
        {"Sign", TranslateUnaryOp<opset8::Sign>},
        {"Softplus", TranslateUnaryOp<opset8::SoftPlus>},
        {"Tan", TranslateUnaryOp<opset8::Tan>},
        {"Tanh", TranslateUnaryOp<opset8::Tanh>},

        // note: BinaryOp translator declaration for each op must to be added in binary_op.cpp file
        {"Add", TranslateBinaryOp<opset8::Add>},
        {"AddV2", TranslateBinaryOp<opset8::Add>},
        {"Equal", TranslateBinaryOp<opset8::Equal>},
        {"FloorMod", TranslateBinaryOp<opset8::FloorMod>},
        {"Greater", TranslateBinaryOp<opset8::Greater>},
        {"GreaterEqual", TranslateBinaryOp<opset8::GreaterEqual>},
        {"Less", TranslateBinaryOp<opset8::Less>},
        {"LessEqual", TranslateBinaryOp<opset8::LessEqual>},
        {"LogicalAnd", TranslateBinaryOp<opset8::LogicalAnd>},
        {"LogicalOr", TranslateBinaryOp<opset8::LogicalOr>},
        {"Maximum", TranslateBinaryOp<opset8::Maximum>},
        {"Minimum", TranslateBinaryOp<opset8::Minimum>},
        {"Mul", TranslateBinaryOp<opset8::Multiply>},
        {"Mod", TranslateBinaryOp<opset8::Mod>},
        {"NotEqual", TranslateBinaryOp<opset8::NotEqual>},
        {"Pow", TranslateBinaryOp<opset8::Power>},
        {"RealDiv", TranslateBinaryOp<opset8::Divide>},
        {"SquaredDifference", TranslateBinaryOp<opset8::SquaredDifference>},
        {"Sub", TranslateBinaryOp<opset8::Subtract>},

        // note: ReduceOp translator declaration for each op must to be added in reduce.cpp file
        {"Any", TranslateDirectReduceOp<opset8::ReduceLogicalOr>},
        {"All", TranslateDirectReduceOp<opset8::ReduceLogicalAnd>},
        {"Max", TranslateDirectReduceOp<opset8::ReduceMax>},
        {"Mean", TranslateDirectReduceOp<opset8::ReduceMean>},
        {"Min", TranslateDirectReduceOp<opset8::ReduceMin>},
        {"Prod", TranslateDirectReduceOp<opset8::ReduceProd>},
        {"Sum", TranslateDirectReduceOp<opset8::ReduceSum>},

        // Separate translators:
        {"AddN", TranslateAddNOp},
        {"ArgMax", TranslateArgMaxOp},
        {"ArgMin", TranslateArgMinOp},
        {"AvgPool", TranslateAvgPoolOp},
        {"BatchToSpaceND", TranslateBatchNDAndSpaceNDOp},
        {"BiasAdd", TranslateBiasAddOp},
        {"Cast", TranslateCastOp},
        {"Concat", TranslateConcatOp},
        {"ConcatV2", TranslateConcatOp},
        {"Const", TranslateConstOp},
        {"Conv2D", TranslateConv2DOp},
        {"Conv2DBackpropInput", TranslateConv2DBackpropInputOp},
        {"Conv3D", TranslateConv3DOp},
        {"Cumsum", TranslateCumsumOp},
        {"DepthToSpace", TranslateDepthToSpaceOp},
        {"DepthwiseConv2dNative", TranslateDepthwiseConv2dNativeOp},
        {"Elu", TranslateEluOp},
        {"ExpandDims", TranslateExpandDimsOp},
        {"Fill", TranslateFillOp},
        {"FloorDiv", TranslateFloorDivOp},
        {"FusedBatchNorm", TranslateFusedBatchNormOp},
        {"FusedBatchNormV2", TranslateFusedBatchNormOp},
        {"FusedBatchNormV3", TranslateFusedBatchNormOp},
        {"Gather", TranslateGatherOp},
        {"GatherV2", TranslateGatherV2Op},
        {"GatherNd", TranslateGatherNdOp},
        {"Identity", TranslateIdentityOp},
        {"IsFinite", TranslateIsFiniteOp},
        {"L2Loss", TranslateL2LossOp},
        {"LogSoftmax", TranslateLogSoftmaxOp},
        {"Log1p", TranslateLog1pOp},
        {"LRN", TranslateLRNOp},
        {"MatMul", TranslateMatMulOp},
        {"MaxPool", TranslateMaxPoolOp},
        {"MaxPool3D", TranslateMaxPoolOp},
        {"MirrorPad", TranslatePadOp},
        {"NonMaxSuppression", TranslateNonMaxSuppressionOp},
        {"NonMaxSuppressionV2", TranslateNonMaxSuppressionOp},
        {"NonMaxSuppressionV3", TranslateNonMaxSuppressionOp},
        {"NonMaxSuppressionV4", TranslateNonMaxSuppressionOp},
        {"NonMaxSuppressionV5", TranslateNonMaxSuppressionOp},
        {"NoOp", NoOp},  // do nothing
        {"OneHot", TranslateOneHotOp},
        {"Pack", TranslatePackOp},
        {"Pad", TranslatePadOp},
        {"PadV2", TranslatePadOp},
        {"Placeholder", PlaceholderOp},
        {"PreventGradient", TranslateIdentityOp},
        {"Range", TranslateRangeOp},
        {"Rank", TranslateRankOp},
        {"RandomUniform", TranslateRandomUniformOp},
        {"Reciprocal", TranslateReciprocalOp},
        {"Relu6", TranslateRelu6Op},
        {"Reshape", TranslateReshapeOp},
        {"_Retval", RetvalOp},
        {"Reverse", TranslateReverseOp},
        {"ReverseV2", TranslateReverseOp},
        {"ResizeBilinear", TranslateInterpolateOp},
        {"ResizeNearestNeighbor", TranslateInterpolateOp},
        {"Roll", TranslateRollOp},
        {"Round", TranslateRoundOp},
        {"Rsqrt", TranslateRsqrtOp},
        {"Select", TranslateSelectOp},
        {"SelectV2", TranslateSelectOp},
        {"Shape", TranslateShapeOp},
        {"Size", TranslateSizeOp},
        {"Slice", TranslateSliceOp},
        {"Snapshot", TranslateIdentityOp},
        {"Softmax", TranslateSoftmaxOp},
        {"SpaceToDepth", TranslateSpaceToDepthOp},
        {"Split", TranslateSplitOp},
        {"SplitV", TranslateSplitVOp},
        {"Sqrt", TranslateSqrtOp},
        {"Square", TranslateSquareOp},
        {"Squeeze", TranslateSqueezeOp},
        {"SpaceToBatchND", TranslateBatchNDAndSpaceNDOp},
        {"StridedSlice", TranslateStridedSliceOp},
        {"Tile", TranslateTileOp},
        {"TopKV2", TranslateTopKV2Op},
        {"Transpose", TranslateTransposeOp},
        {"Unpack", TranslateUnpackOp},
        {"Where", TranslateWhereOp},
        {"Xdivy", TranslateXdivyOp},
        {"ZerosLike", TranslateZerosLikeOp},

        {"_FusedConv2D", TranslateFusedConv2DOp},
        {"_FusedMatMul", TranslateFusedMatMulOp},
        {"_FusedBatchNormEx", TranslateFusedBatchNormOp},
        {"_FusedDepthwiseConv2dNative", TranslateDepthwiseConv2dNativeOp},
        {"FakeQuantWithMinMaxVars", TranslateFakeQuantWithMinMaxVarsOp},
        {"CropAndResize", TranslateCropAndResizeOp},
        // {"_Arg", ArgOp}, // should be registered as an extension in OVTF
    };
};
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph