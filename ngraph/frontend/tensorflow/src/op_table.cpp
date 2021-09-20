// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"

#include <tensorflow_frontend/node_context.hpp>

using namespace std;
using namespace ngraph;
using namespace frontend::tensorflow::detail;

namespace tensorflow {
namespace ngraph_bridge {
#define OP_CONVERTER(op) NamedOutputs op(const NodeContext& node)
#define OP_T_CONVERTER(op) \
    template <class T>     \
    NamedOutputs op(const NodeContext& node)
#define OP_UINT_CONVERTER(op) \
    template <unsigned int T> \
    NamedOutputs op(const NodeContext& node)

OP_T_CONVERTER(TranslateUnaryOp);
OP_T_CONVERTER(TranslateBinaryOp);
OP_T_CONVERTER(TranslateDirectReduceOp);

OP_UINT_CONVERTER(TranslateMaxPoolOp);

OP_CONVERTER(TranslateAddNOp);
OP_CONVERTER(TranslateArgMaxOp);
OP_CONVERTER(TranslateArgMinOp);
OP_CONVERTER(TranslateAvgPoolOp);
OP_CONVERTER(TranslateBiasAddOp);
OP_CONVERTER(TranslateCastOp);
OP_CONVERTER(TranslateConcatV2Op);
OP_CONVERTER(TranslateConstOp);
OP_CONVERTER(TranslateConv2DOp);
OP_CONVERTER(TranslateConv2DBackpropInputOp);
OP_CONVERTER(TranslateConv3DOp);
OP_CONVERTER(TranslateCumsumOp);
OP_CONVERTER(TranslateDepthToSpaceOp);
OP_CONVERTER(TranslateDepthwiseConv2dNativeOp);
OP_CONVERTER(TranslateExpandDimsOp);
OP_CONVERTER(TranslateFillOp);
OP_CONVERTER(TranslateFloorDivOp);
OP_CONVERTER(TranslateFusedBatchNormOp);
OP_CONVERTER(TranslateGatherOp);
OP_CONVERTER(TranslateGatherV2Op);
OP_CONVERTER(TranslateFusedConv2DOp);
OP_CONVERTER(TranslateFusedMatMulOp);
OP_CONVERTER(TranslateIdentityOp);
// OP_CONVERTER(TranslateIsFiniteOp);
// OP_CONVERTER(TranslateL2LossOp);
OP_CONVERTER(TranslateLogSoftmaxOp);
// OP_CONVERTER(TranslateLog1pOp);
// OP_CONVERTER(TranslateLRNOp);
// OP_CONVERTER(TranslateMatMulOp);
OP_CONVERTER(TranslateNonMaxSuppressionV2Op);
OP_CONVERTER(TranslatePadOp);
OP_CONVERTER(PlaceholderOp);
OP_CONVERTER(NoOp);
// OP_CONVERTER(TranslateOneHotOp);
// OP_CONVERTER(TranslatePackOp);
OP_CONVERTER(TranslateRangeOp);
OP_CONVERTER(TranslateRankOp);
// OP_CONVERTER(TranslateRelu6Op);
// OP_CONVERTER(TranslateReciprocalOp);
// OP_CONVERTER(TranslateReshapeOp);
OP_CONVERTER(RetvalOp);
// OP_CONVERTER(TranslateRsqrtOp);
// OP_CONVERTER(TranslateSelectOp);
// OP_CONVERTER(TranslateShapeOp);
// OP_CONVERTER(TranslateSizeOp);
// OP_CONVERTER(TranslateSliceOp);
// OP_CONVERTER(transpose2);
OP_CONVERTER(TranslateSoftmaxOp);
// OP_CONVERTER(TranslateSpaceToDepthOp);
// OP_CONVERTER(TranslateSplitOp);
// OP_CONVERTER(TranslateSplitOp);
OP_CONVERTER(TranslateSqueezeOp);
// OP_CONVERTER(TranslateStridedSliceOp);
// OP_CONVERTER(TranslateTileOp);
// OP_CONVERTER(TranslateTopKV2Op);
// OP_CONVERTER(TranslateTransposeOp);
// OP_CONVERTER(TranslateUnpackOp);
// OP_CONVERTER(TranslateWhereOp);
// OP_CONVERTER(TranslateXdivyOp);
// OP_CONVERTER(TranslateZerosLikeOp);
}  // namespace ngraph_bridge
}  // namespace tensorflow

namespace tensorflow {
namespace ngraph_bridge {
const std::map<const std::string, CreatorFunction> TRANSLATE_OP_MAP{
    {"Abs", TranslateUnaryOp<opset::Abs>},
    {"Acos", TranslateUnaryOp<opset::Acos>},
    {"Acosh", TranslateUnaryOp<opset::Acosh>},
    {"Add", TranslateBinaryOp<opset::Add>},
    {"AddN", TranslateAddNOp},
    {"AddV2", TranslateBinaryOp<opset::Add>},
    {"Any", TranslateDirectReduceOp<opset::ReduceLogicalOr>},
    {"All", TranslateDirectReduceOp<opset::ReduceLogicalAnd>},
    {"ArgMax", TranslateArgMaxOp},
    {"ArgMin", TranslateArgMinOp},
    {"Asin", TranslateUnaryOp<opset::Asin>},
    {"Asinh", TranslateUnaryOp<opset::Asinh>},
    {"Atan", TranslateUnaryOp<opset::Atan>},
    {"Atanh", TranslateUnaryOp<opset::Atanh>},
    {"AvgPool", TranslateAvgPoolOp},
    {"BiasAdd", TranslateBiasAddOp},
    {"Cast", TranslateCastOp},
    {"Ceil", TranslateUnaryOp<opset::Ceiling>},
    {"ConcatV2", TranslateConcatV2Op},
    {"Const", TranslateConstOp},
    {"Conv2D", TranslateConv2DOp},
    {"Conv2DBackpropInput", TranslateConv2DBackpropInputOp},
    {"Conv3D", TranslateConv3DOp},
    {"Cos", TranslateUnaryOp<opset::Cos>},
    {"Cosh", TranslateUnaryOp<opset::Cosh>},
    {"Cumsum", TranslateCumsumOp},
    {"DepthToSpace", TranslateDepthToSpaceOp},
    {"DepthwiseConv2dNative", TranslateDepthwiseConv2dNativeOp},
    {"Equal", TranslateBinaryOp<opset::Equal>},
    {"Exp", TranslateUnaryOp<opset::Exp>},
    {"ExpandDims", TranslateExpandDimsOp},
    {"Fill", TranslateFillOp},
    {"Floor", TranslateUnaryOp<opset::Floor>},
    {"FloorDiv", TranslateFloorDivOp},
    {"FloorMod", TranslateBinaryOp<opset::FloorMod>},
    {"FusedBatchNorm", TranslateFusedBatchNormOp},
    {"FusedBatchNormV2", TranslateFusedBatchNormOp},
    {"FusedBatchNormV3", TranslateFusedBatchNormOp},
    {"Gather", TranslateGatherOp},
    {"GatherV2", TranslateGatherV2Op},
    {"_FusedConv2D", TranslateFusedConv2DOp},
    {"_FusedMatMul", TranslateFusedMatMulOp},
    {"Greater", TranslateBinaryOp<opset::Greater>},
    {"GreaterEqual", TranslateBinaryOp<opset::GreaterEqual>},
    {"Identity", TranslateIdentityOp},
    //{"IsFinite", TranslateIsFiniteOp},
    //{"L2Loss", TranslateL2LossOp},
    {"LogSoftmax", TranslateLogSoftmaxOp},
    {"Less", TranslateBinaryOp<opset::Less>},
    {"LessEqual", TranslateBinaryOp<opset::LessEqual>},
    {"Log", TranslateUnaryOp<opset::Log>},
    //{"Log1p", TranslateLog1pOp},
    {"LogicalAnd", TranslateBinaryOp<opset::LogicalAnd>},
    {"LogicalNot", TranslateUnaryOp<opset::LogicalNot>},
    {"LogicalOr", TranslateBinaryOp<opset::LogicalOr>},
    //{"LRN", TranslateLRNOp},
    //{"MatMul", TranslateMatMulOp},
    {"Max", TranslateDirectReduceOp<opset::ReduceMax>},
    {"Maximum", TranslateBinaryOp<opset::Maximum>},
    {"MaxPool", TranslateMaxPoolOp<2>},
    {"MaxPool3D", TranslateMaxPoolOp<3>},
    //{"NonMaxSuppressionV2", TranslateNonMaxSuppressionV2Op},
    {"Mean", TranslateDirectReduceOp<opset::ReduceMean>},
    {"Min", TranslateDirectReduceOp<opset::ReduceMin>},
    {"Minimum", TranslateBinaryOp<opset::Minimum>},
    {"MirrorPad", TranslatePadOp},
    {"Mul", TranslateBinaryOp<opset::Multiply>},
    {"Mod", TranslateBinaryOp<opset::Mod>},
    {"Neg", TranslateUnaryOp<opset::Negative>},
    {"NotEqual", TranslateBinaryOp<opset::NotEqual>},
    // Do nothing! NoOps sometimes get placed on nGraph for bureaucratic
    // reasons, but they have no data flow inputs or outputs.
    {"NoOp", NoOp},
    //{"OneHot", TranslateOneHotOp},
    //{"Pack", TranslatePackOp},
    {"Pad", TranslatePadOp},
    {"PadV2", TranslatePadOp},
    //{"_Arg", ArgOp}, // should be registered as an extension in OVTF
    {"Placeholder", PlaceholderOp},
    {"Pow", TranslateBinaryOp<opset::Power>},
    // PreventGradient is just Identity in dataflow terms, so reuse that.
    {"PreventGradient", TranslateIdentityOp},
    {"Prod", TranslateDirectReduceOp<opset::ReduceProd>},
    //{"Range", TranslateRangeOp},
    //{"Rank", TranslateRankOp},
    {"RealDiv", TranslateBinaryOp<opset::Divide>},
    //{"Reciprocal", TranslateReciprocalOp},
    {"Relu", TranslateUnaryOp<opset::Relu>},
    //{"Relu6", TranslateRelu6Op},
    //{"Reshape", TranslateReshapeOp},
    {"_Retval", RetvalOp},
    //{"Rsqrt", TranslateRsqrtOp},
    //{"Select", TranslateSelectOp},
    //{"SelectV2", TranslateSelectOp},
    //{"Shape", TranslateShapeOp},
    {"Sigmoid", TranslateUnaryOp<opset::Sigmoid>},
    {"Sin", TranslateUnaryOp<opset::Sin>},
    {"Sinh", TranslateUnaryOp<opset::Sinh>},
    //{"Size", TranslateSizeOp},
    {"Sign", TranslateUnaryOp<opset::Sign>},
    //{"Slice", TranslateSliceOp},
    //{"Snapshot", TranslateIdentityOp},
    {"Softmax", TranslateSoftmaxOp},
    {"Softplus", TranslateUnaryOp<opset::SoftPlus>},
    //{"SpaceToDepth", TranslateSpaceToDepthOp},
    //{"Split", TranslateSplitOp},
    //{"SplitV", TranslateSplitVOp},
    {"Sqrt", TranslateUnaryOp<opset::Sqrt>},
    //{"Square", TranslateSquareOp},
    {"SquaredDifference", TranslateBinaryOp<opset::SquaredDifference>},
    {"Squeeze", TranslateSqueezeOp},
    //{"StridedSlice", TranslateStridedSliceOp},
    {"Sub", TranslateBinaryOp<opset::Subtract>},
    {"Sum", TranslateDirectReduceOp<opset::ReduceSum>},
    {"Tan", TranslateUnaryOp<opset::Tan>},
    {"Tanh", TranslateUnaryOp<opset::Tanh>},
    //{"Tile", TranslateTileOp},
    //{"TopKV2", TranslateTopKV2Op},
    //{"Transpose", TranslateTransposeOp},
    //{"Unpack", TranslateUnpackOp},
    //{"Where", TranslateWhereOp},
    //{"Xdivy", TranslateXdivyOp},
    //{"ZerosLike", TranslateZerosLikeOp},
};
}  // namespace ngraph_bridge
}  // namespace tensorflow