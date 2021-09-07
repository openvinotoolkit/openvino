// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef OPENVINO_OP
#    warning "OPENVINO_OP not defined"
#    define OPENVINO_OP(x, y)
#endif

OPENVINO_OP(Abs, ngraph::op::v0)
OPENVINO_OP(Acos, ngraph::op::v0)
OPENVINO_OP(Add, ngraph::op::v1)
OPENVINO_OP(Asin, ngraph::op::v0)
OPENVINO_OP(Atan, ngraph::op::v0)
OPENVINO_OP(AvgPool, ngraph::op::v1)
OPENVINO_OP(BatchNormInference, ngraph::op::v0)
OPENVINO_OP(BinaryConvolution, ngraph::op::v1)
OPENVINO_OP(Broadcast, ngraph::op::v1)
OPENVINO_OP(CTCGreedyDecoder, ngraph::op::v0)
OPENVINO_OP(Ceiling, ngraph::op::v0)
OPENVINO_OP(Clamp, ngraph::op::v0)
OPENVINO_OP(Concat, ngraph::op::v0)
OPENVINO_OP(Constant, ngraph::op)
OPENVINO_OP(Convert, ngraph::op::v0)
OPENVINO_OP(ConvertLike, ngraph::op::v1)
OPENVINO_OP(Convolution, ngraph::op::v1)
OPENVINO_OP(ConvolutionBackpropData, ngraph::op::v1)
OPENVINO_OP(Cos, ngraph::op::v0)
OPENVINO_OP(Cosh, ngraph::op::v0)
OPENVINO_OP(DeformableConvolution, ngraph::op::v1)
OPENVINO_OP(DeformablePSROIPooling, ngraph::op::v1)
OPENVINO_OP(DepthToSpace, ngraph::op::v0)
OPENVINO_OP(DetectionOutput, ngraph::op::v0)
OPENVINO_OP(Divide, ngraph::op::v1)
OPENVINO_OP(Elu, ngraph::op::v0)
OPENVINO_OP(Erf, ngraph::op::v0)
OPENVINO_OP(Equal, ngraph::op::v1)
OPENVINO_OP(Exp, ngraph::op::v0)
OPENVINO_OP(FakeQuantize, ngraph::op::v0)
OPENVINO_OP(Floor, ngraph::op::v0)
OPENVINO_OP(FloorMod, ngraph::op::v1)
OPENVINO_OP(Gather, ngraph::op::v1)
OPENVINO_OP(GatherTree, ngraph::op::v1)
OPENVINO_OP(Greater, ngraph::op::v1)
OPENVINO_OP(GreaterEqual, ngraph::op::v1)
OPENVINO_OP(GroupConvolution, ngraph::op::v1)
OPENVINO_OP(GroupConvolutionBackpropData, ngraph::op::v1)
OPENVINO_OP(GRN, ngraph::op::v0)
OPENVINO_OP(HardSigmoid, ngraph::op::v0)
OPENVINO_OP(Interpolate, ngraph::op::v0)
OPENVINO_OP(Less, ngraph::op::v1)
OPENVINO_OP(LessEqual, ngraph::op::v1)
OPENVINO_OP(Log, ngraph::op::v0)
OPENVINO_OP(LogicalAnd, ngraph::op::v1)
OPENVINO_OP(LogicalNot, ngraph::op::v1)
OPENVINO_OP(LogicalOr, ngraph::op::v1)
OPENVINO_OP(LogicalXor, ngraph::op::v1)
OPENVINO_OP(LRN, ngraph::op::v0)
OPENVINO_OP(LSTMCell, ngraph::op::v0)
OPENVINO_OP(LSTMSequence, ngraph::op::v0)
OPENVINO_OP(MatMul, ngraph::op::v0)
OPENVINO_OP(MaxPool, ngraph::op::v1)
OPENVINO_OP(Maximum, ngraph::op::v1)
OPENVINO_OP(Minimum, ngraph::op::v1)
OPENVINO_OP(Mod, ngraph::op::v1)
OPENVINO_OP(Multiply, ngraph::op::v1)

OPENVINO_OP(MVN, ngraph::op::v0)  // Missing in opset1

OPENVINO_OP(Negative, ngraph::op::v0)
OPENVINO_OP(NonMaxSuppression, ngraph::op::v1)
OPENVINO_OP(NormalizeL2, ngraph::op::v0)
OPENVINO_OP(NotEqual, ngraph::op::v1)
OPENVINO_OP(OneHot, ngraph::op::v1)
OPENVINO_OP(PRelu, ngraph::op::v0)
OPENVINO_OP(PSROIPooling, ngraph::op::v0)
OPENVINO_OP(Pad, ngraph::op::v1)
OPENVINO_OP(Parameter, ngraph::op::v0)
OPENVINO_OP(Power, ngraph::op::v1)
OPENVINO_OP(PriorBox, ngraph::op::v0)
OPENVINO_OP(PriorBoxClustered, ngraph::op::v0)
OPENVINO_OP(Proposal, ngraph::op::v0)
OPENVINO_OP(Range, ngraph::op::v0)
OPENVINO_OP(Relu, ngraph::op::v0)
OPENVINO_OP(ReduceMax, ngraph::op::v1)
OPENVINO_OP(ReduceLogicalAnd, ngraph::op::v1)
OPENVINO_OP(ReduceLogicalOr, ngraph::op::v1)
OPENVINO_OP(ReduceMean, ngraph::op::v1)
OPENVINO_OP(ReduceMin, ngraph::op::v1)
OPENVINO_OP(ReduceProd, ngraph::op::v1)
OPENVINO_OP(ReduceSum, ngraph::op::v1)
OPENVINO_OP(RegionYolo, ngraph::op::v0)

OPENVINO_OP(ReorgYolo, ngraph::op::v0)  // Missing in opset1

OPENVINO_OP(Reshape, ngraph::op::v1)
OPENVINO_OP(Result, ngraph::op::v0)

// Moved out of opset2, it was added to opset1 by mistake
// OPENVINO_OP(Reverse, ngraph::op::v1)

OPENVINO_OP(ReverseSequence, ngraph::op::v0)

// Moved out of opset2, it was added to opset1 by mistake
// OPENVINO_OP(RNNCell, ngraph::op::v0)

OPENVINO_OP(ROIPooling, ngraph::op::v0)  // Missing in opset1

OPENVINO_OP(Select, ngraph::op::v1)
OPENVINO_OP(Selu, ngraph::op::v0)
OPENVINO_OP(ShapeOf, ngraph::op::v0)

// Moved out of opset2, it was added to opset1 by mistake
// OPENVINO_OP(ShuffleChannels, ngraph::op::v0)

OPENVINO_OP(Sign, ngraph::op::v0)
OPENVINO_OP(Sigmoid, ngraph::op::v0)
OPENVINO_OP(Sin, ngraph::op::v0)
OPENVINO_OP(Sinh, ngraph::op::v0)
OPENVINO_OP(Softmax, ngraph::op::v1)
OPENVINO_OP(Sqrt, ngraph::op::v0)
OPENVINO_OP(SpaceToDepth, ngraph::op::v0)
OPENVINO_OP(Split, ngraph::op::v1)
OPENVINO_OP(SquaredDifference, ngraph::op::v0)
OPENVINO_OP(Squeeze, ngraph::op::v0)
OPENVINO_OP(StridedSlice, ngraph::op::v1)
OPENVINO_OP(Subtract, ngraph::op::v1)
OPENVINO_OP(Tan, ngraph::op::v0)
OPENVINO_OP(Tanh, ngraph::op::v0)
OPENVINO_OP(TensorIterator, ngraph::op::v0)
OPENVINO_OP(Tile, ngraph::op::v0)
OPENVINO_OP(TopK, ngraph::op::v1)
OPENVINO_OP(Transpose, ngraph::op::v1)
OPENVINO_OP(Unsqueeze, ngraph::op::v0)
OPENVINO_OP(VariadicSplit, ngraph::op::v1)

// Moved out of opset2, it was added to opset1 by mistake
// OPENVINO_OP(Xor, ngraph::op::v0)

// New operations added in opset2
OPENVINO_OP(Gelu, ngraph::op::v0)
OPENVINO_OP(BatchToSpace, ngraph::op::v1)
OPENVINO_OP(SpaceToBatch, ngraph::op::v1)
