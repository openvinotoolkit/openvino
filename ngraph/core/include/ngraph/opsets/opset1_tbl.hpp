// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This collection contains one entry for each op. If an op is added it must be
// added to this list.
//
// In order to use this list you want to define a macro named exactly NGRAPH_OP
// When you are done you should undef the macro
// As an example if you wanted to make a list of all op names as strings you could do this:
//
// #define NGRAPH_OP(a,b) #a,
// std::vector<std::string> op_names{
// #include "this include file name"
// };
// #undef NGRAPH_OP
//
// This sample expands to a list like this:
// "Abs",
// "Acos",
// ...
//
// #define NGRAPH_OP(a,b) b::a,
// std::vector<std::string> op_names{
// #include "this include file name"
// };
// #undef NGRAPH_OP
//
// This sample expands to a list like this:
// ov::op::Abs,
// ov::op::Acos,
// ...
//
// It's that easy. You can use this for fun and profit.

#ifndef NGRAPH_OP
#warning "NGRAPH_OP not defined"
#define NGRAPH_OP(x, y)
#endif

NGRAPH_OP(Abs, ov::op::v0)
NGRAPH_OP(Acos, ov::op::v0)
NGRAPH_OP(Add, ov::op::v1)
NGRAPH_OP(Asin, ov::op::v0)
NGRAPH_OP(Atan, ov::op::v0)
NGRAPH_OP(AvgPool, ov::op::v1)
NGRAPH_OP(BatchNormInference, ov::op::v0)
NGRAPH_OP(BinaryConvolution, ov::op::v1)
NGRAPH_OP(Broadcast, ov::op::v1)
NGRAPH_OP(CTCGreedyDecoder, ov::op::v0)
NGRAPH_OP(Ceiling, ov::op::v0)
NGRAPH_OP(Clamp, ov::op::v0)
NGRAPH_OP(Concat, ov::op::v0)
NGRAPH_OP(Constant, ov::op)
NGRAPH_OP(Convert, ov::op::v0)
NGRAPH_OP(ConvertLike, ov::op::v1)
NGRAPH_OP(Convolution, ov::op::v1)
NGRAPH_OP(ConvolutionBackpropData, ov::op::v1)
NGRAPH_OP(Cos, ov::op::v0)
NGRAPH_OP(Cosh, ov::op::v0)
NGRAPH_OP(DeformableConvolution, ov::op::v1)
NGRAPH_OP(DeformablePSROIPooling, ov::op::v1)
NGRAPH_OP(DepthToSpace, ov::op::v0)
NGRAPH_OP(DetectionOutput, ov::op::v0)
NGRAPH_OP(Divide, ov::op::v1)
NGRAPH_OP(Elu, ov::op::v0)
NGRAPH_OP(Erf, ov::op::v0)
NGRAPH_OP(Equal, ov::op::v1)
NGRAPH_OP(Exp, ov::op::v0)
NGRAPH_OP(FakeQuantize, ov::op::v0)
NGRAPH_OP(Floor, ov::op::v0)
NGRAPH_OP(FloorMod, ov::op::v1)
NGRAPH_OP(Gather, ov::op::v1)
NGRAPH_OP(GatherTree, ov::op::v1)
NGRAPH_OP(Greater, ov::op::v1)
NGRAPH_OP(GreaterEqual, ov::op::v1)
NGRAPH_OP(GroupConvolution, ov::op::v1)
NGRAPH_OP(GroupConvolutionBackpropData, ov::op::v1)
NGRAPH_OP(GRN, ov::op::v0)
NGRAPH_OP(HardSigmoid, ov::op::v0)
NGRAPH_OP(Interpolate, ov::op::v0)
NGRAPH_OP(Less, ov::op::v1)
NGRAPH_OP(LessEqual, ov::op::v1)
NGRAPH_OP(Log, ov::op::v0)
NGRAPH_OP(LogicalAnd, ov::op::v1)
NGRAPH_OP(LogicalNot, ov::op::v1)
NGRAPH_OP(LogicalOr, ov::op::v1)
NGRAPH_OP(LogicalXor, ov::op::v1)
NGRAPH_OP(LRN, ov::op::v0)
NGRAPH_OP(LSTMCell, ov::op::v0)
NGRAPH_OP(LSTMSequence, ov::op::v0)
NGRAPH_OP(MatMul, ov::op::v0)
NGRAPH_OP(MaxPool, ov::op::v1)
NGRAPH_OP(Maximum, ov::op::v1)
NGRAPH_OP(Minimum, ov::op::v1)
NGRAPH_OP(Mod, ov::op::v1)
NGRAPH_OP(Multiply, ov::op::v1)
NGRAPH_OP(Negative, ov::op::v0)
NGRAPH_OP(NonMaxSuppression, ov::op::v1)
NGRAPH_OP(NormalizeL2, ov::op::v0)
NGRAPH_OP(NotEqual, ov::op::v1)
NGRAPH_OP(OneHot, ov::op::v1)
NGRAPH_OP(PRelu, ov::op::v0)
NGRAPH_OP(PSROIPooling, ov::op::v0)
NGRAPH_OP(Pad, ov::op::v1)
NGRAPH_OP(Parameter, ov::op::v0)
NGRAPH_OP(Power, ov::op::v1)
NGRAPH_OP(PriorBox, ov::op::v0)
NGRAPH_OP(PriorBoxClustered, ov::op::v0)
NGRAPH_OP(Proposal, ov::op::v0)
NGRAPH_OP(Range, ov::op::v0)
NGRAPH_OP(Relu, ov::op::v0)
NGRAPH_OP(ReduceMax, ov::op::v1)
NGRAPH_OP(ReduceLogicalAnd, ov::op::v1)
NGRAPH_OP(ReduceLogicalOr, ov::op::v1)
NGRAPH_OP(ReduceMean, ov::op::v1)
NGRAPH_OP(ReduceMin, ov::op::v1)
NGRAPH_OP(ReduceProd, ov::op::v1)
NGRAPH_OP(ReduceSum, ov::op::v1)
NGRAPH_OP(RegionYolo, ov::op::v0)
NGRAPH_OP(Reshape, ov::op::v1)
NGRAPH_OP(Result, ov::op::v0)
NGRAPH_OP(Reverse, ov::op::v1)
NGRAPH_OP(ReverseSequence, ov::op::v0)
NGRAPH_OP(RNNCell, ov::op::v0)
NGRAPH_OP(Select, ov::op::v1)
NGRAPH_OP(Selu, ov::op::v0)
NGRAPH_OP(ShapeOf, ov::op::v0)
NGRAPH_OP(ShuffleChannels, ov::op::v0)
NGRAPH_OP(Sign, ov::op::v0)
NGRAPH_OP(Sigmoid, ov::op::v0)
NGRAPH_OP(Sin, ov::op::v0)
NGRAPH_OP(Sinh, ov::op::v0)
NGRAPH_OP(Softmax, ov::op::v1)
NGRAPH_OP(Sqrt, ov::op::v0)
NGRAPH_OP(SpaceToDepth, ov::op::v0)
NGRAPH_OP(Split, ov::op::v1)
NGRAPH_OP(SquaredDifference, ov::op::v0)
NGRAPH_OP(Squeeze, ov::op::v0)
NGRAPH_OP(StridedSlice, ov::op::v1)
NGRAPH_OP(Subtract, ov::op::v1)
NGRAPH_OP(Tan, ov::op::v0)
NGRAPH_OP(Tanh, ov::op::v0)
NGRAPH_OP(TensorIterator, ov::op::v0)
NGRAPH_OP(Tile, ov::op::v0)
NGRAPH_OP(TopK, ov::op::v1)
NGRAPH_OP(Transpose, ov::op::v1)
NGRAPH_OP(Unsqueeze, ov::op::v0)
NGRAPH_OP(VariadicSplit, ov::op::v1)
NGRAPH_OP(Xor, ov::op::v0)
