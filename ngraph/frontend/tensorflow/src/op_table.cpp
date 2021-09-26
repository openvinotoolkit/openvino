// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include <tensorflow_frontend/node_context.hpp>

using namespace std;
using namespace ngraph;
using namespace frontend::tensorflow::detail;
using namespace ngraph::frontend::tf;

namespace tensorflow {
namespace ngraph_bridge {
#define OP_CONVERTER(op) NamedOutputs op(const NodeContext& node)
#define OP_T_CONVERTER(op) \
    template <class T>     \
    NamedOutputs op(const NodeContext& node)

OP_T_CONVERTER(TranslateUnaryOp);
OP_T_CONVERTER(TranslateBinaryOp);
OP_T_CONVERTER(TranslateDirectReduceOp);

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
OP_CONVERTER(TranslateMaxPoolOp);
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
void SetTracingInfo(const std::string& op_name, const ngraph::Output<ngraph::Node> ng_node) {
    auto node = ng_node.get_node_shared_ptr();
    node->set_friendly_name(op_name);
    node->add_provenance_tag(op_name);
}

const std::map<const std::string, const CreatorFunction> get_supported_ops() {
    return {
        // note: UnaryOp translator declaration for each op must to be added in unary_op.cpp file
        {"Abs", TranslateUnaryOp<opset::Abs>},
        {"Acos", TranslateUnaryOp<opset::Acos>},
        {"Acosh", TranslateUnaryOp<opset::Acosh>},
        {"Asin", TranslateUnaryOp<opset::Asin>},
        {"Asinh", TranslateUnaryOp<opset::Asinh>},
        {"Atan", TranslateUnaryOp<opset::Atan>},
        {"Atanh", TranslateUnaryOp<opset::Atanh>},
        {"Ceil", TranslateUnaryOp<opset::Ceiling>},
        {"Cos", TranslateUnaryOp<opset::Cos>},
        {"Cosh", TranslateUnaryOp<opset::Cosh>},
        {"Exp", TranslateUnaryOp<opset::Exp>},
        {"Floor", TranslateUnaryOp<opset::Floor>},
        {"Log", TranslateUnaryOp<opset::Log>},
        {"LogicalNot", TranslateUnaryOp<opset::LogicalNot>},
        {"Neg", TranslateUnaryOp<opset::Negative>},
        {"Relu", TranslateUnaryOp<opset::Relu>},
        {"Sigmoid", TranslateUnaryOp<opset::Sigmoid>},
        {"Sin", TranslateUnaryOp<opset::Sin>},
        {"Sinh", TranslateUnaryOp<opset::Sinh>},
        {"Sign", TranslateUnaryOp<opset::Sign>},
        {"Softplus", TranslateUnaryOp<opset::SoftPlus>},
        {"Sqrt", TranslateUnaryOp<opset::Sqrt>},
        {"Tan", TranslateUnaryOp<opset::Tan>},
        {"Tanh", TranslateUnaryOp<opset::Tanh>},

        // note: BinaryOp translator declaration for each op  must to be added in binary_op.cpp file
        {"Add", TranslateBinaryOp<opset::Add>},
        {"AddV2", TranslateBinaryOp<opset::Add>},
        {"Equal", TranslateBinaryOp<opset::Equal>},
        {"FloorMod", TranslateBinaryOp<opset::FloorMod>},
        {"Greater", TranslateBinaryOp<opset::Greater>},
        {"GreaterEqual", TranslateBinaryOp<opset::GreaterEqual>},
        {"Less", TranslateBinaryOp<opset::Less>},
        {"LessEqual", TranslateBinaryOp<opset::LessEqual>},
        {"LogicalAnd", TranslateBinaryOp<opset::LogicalAnd>},
        {"LogicalOr", TranslateBinaryOp<opset::LogicalOr>},
        {"Maximum", TranslateBinaryOp<opset::Maximum>},
        {"Minimum", TranslateBinaryOp<opset::Minimum>},
        {"Mul", TranslateBinaryOp<opset::Multiply>},
        {"Mod", TranslateBinaryOp<opset::Mod>},
        {"NotEqual", TranslateBinaryOp<opset::NotEqual>},
        {"Pow", TranslateBinaryOp<opset::Power>},
        {"RealDiv", TranslateBinaryOp<opset::Divide>},
        {"SquaredDifference", TranslateBinaryOp<opset::SquaredDifference>},
        {"Sub", TranslateBinaryOp<opset::Subtract>},

        // note: ReduceOp translator declaration for each op must to be added in reduce.cpp file
        {"Any", TranslateDirectReduceOp<opset::ReduceLogicalOr>},
        {"All", TranslateDirectReduceOp<opset::ReduceLogicalAnd>},
        {"Max", TranslateDirectReduceOp<opset::ReduceMax>},
        {"Mean", TranslateDirectReduceOp<opset::ReduceMean>},
        {"Min", TranslateDirectReduceOp<opset::ReduceMin>},
        {"Prod", TranslateDirectReduceOp<opset::ReduceProd>},
        {"Sum", TranslateDirectReduceOp<opset::ReduceSum>},

        // Separate translators:
        {"AddN", TranslateAddNOp},
        {"ArgMax", TranslateArgMaxOp},
        {"ArgMin", TranslateArgMinOp},
        {"AvgPool", TranslateAvgPoolOp},
        {"BiasAdd", TranslateBiasAddOp},
        {"Cast", TranslateCastOp},
        {"ConcatV2", TranslateConcatV2Op},
        {"Const", TranslateConstOp},
        {"Conv2D", TranslateConv2DOp},
        {"Conv2DBackpropInput", TranslateConv2DBackpropInputOp},
        {"Conv3D", TranslateConv3DOp},
        {"Cumsum", TranslateCumsumOp},
        {"DepthToSpace", TranslateDepthToSpaceOp},
        {"DepthwiseConv2dNative", TranslateDepthwiseConv2dNativeOp},
        {"ExpandDims", TranslateExpandDimsOp},
        {"Fill", TranslateFillOp},
        {"FloorDiv", TranslateFloorDivOp},
        {"FusedBatchNorm", TranslateFusedBatchNormOp},
        {"FusedBatchNormV2", TranslateFusedBatchNormOp},
        {"FusedBatchNormV3", TranslateFusedBatchNormOp},
        {"Gather", TranslateGatherOp},
        {"GatherV2", TranslateGatherV2Op},
        {"_FusedConv2D", TranslateFusedConv2DOp},
        {"_FusedMatMul", TranslateFusedMatMulOp},
        {"Identity", TranslateIdentityOp},
        //{"IsFinite", TranslateIsFiniteOp},
        //{"L2Loss", TranslateL2LossOp},
        {"LogSoftmax", TranslateLogSoftmaxOp},
        //{"Log1p", TranslateLog1pOp},
        //{"LRN", TranslateLRNOp},
        //{"MatMul", TranslateMatMulOp},
        {"MaxPool", TranslateMaxPoolOp},
        {"MaxPool3D", TranslateMaxPoolOp},
        //{"NonMaxSuppressionV2", TranslateNonMaxSuppressionV2Op},
        {"MirrorPad", TranslatePadOp},
        {"NoOp", NoOp},  // do nothing
                         //{"OneHot", TranslateOneHotOp},
                         //{"Pack", TranslatePackOp},
        {"Pad", TranslatePadOp},
        {"PadV2", TranslatePadOp},
        //{"_Arg", ArgOp}, // should be registered as an extension in OVTF
        {"Placeholder", PlaceholderOp},
        // PreventGradient is just Identity in dataflow terms, so reuse that.
        {"PreventGradient", TranslateIdentityOp},
        //{"Range", TranslateRangeOp},
        //{"Rank", TranslateRankOp},
        //{"Reciprocal", TranslateReciprocalOp},
        //{"Relu6", TranslateRelu6Op},
        //{"Reshape", TranslateReshapeOp},
        {"_Retval", RetvalOp},
        //{"Rsqrt", TranslateRsqrtOp},
        //{"Select", TranslateSelectOp},
        //{"SelectV2", TranslateSelectOp},
        //{"Shape", TranslateShapeOp},
        //{"Size", TranslateSizeOp},
        //{"Slice", TranslateSliceOp},
        //{"Snapshot", TranslateIdentityOp},
        {"Softmax", TranslateSoftmaxOp},
        //{"SpaceToDepth", TranslateSpaceToDepthOp},
        //{"Split", TranslateSplitOp},
        //{"SplitV", TranslateSplitVOp},
        //{"Square", TranslateSquareOp},
        {"Squeeze", TranslateSqueezeOp},
        //{"StridedSlice", TranslateStridedSliceOp},
        //{"Tile", TranslateTileOp},
        //{"TopKV2", TranslateTopKV2Op},
        //{"Transpose", TranslateTransposeOp},
        //{"Unpack", TranslateUnpackOp},
        //{"Where", TranslateWhereOp},
        //{"Xdivy", TranslateXdivyOp},
        //{"ZerosLike", TranslateZerosLikeOp},
    };
};
}  // namespace ngraph_bridge
}  // namespace tensorflow