// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef OPENVINO_OP
#    warning "OPENVINO_OP not defined"
#    define OPENVINO_OP(x, y)
#endif

OPENVINO_OP(Abs, ov::op::v0)
OPENVINO_OP(Acos, ov::op::v0)
OPENVINO_OP(Add, ov::op::v1)
OPENVINO_OP(Asin, ov::op::v0)
OPENVINO_OP(Atan, ov::op::v0)
OPENVINO_OP(AvgPool, ov::op::v1)
OPENVINO_OP(BatchNormInference, ov::op::v0)
OPENVINO_OP(BinaryConvolution, ov::op::v1)
OPENVINO_OP(Broadcast, ov::op::v1)
OPENVINO_OP(CTCGreedyDecoder, ov::op::v0)
OPENVINO_OP(Ceiling, ov::op::v0)
OPENVINO_OP(Clamp, ov::op::v0)
OPENVINO_OP(Concat, ov::op::v0)
OPENVINO_OP(Constant, ov::op::v0)
OPENVINO_OP(Convert, ov::op::v0)
OPENVINO_OP(ConvertLike, ov::op::v1)
OPENVINO_OP(Convolution, ov::op::v1)
OPENVINO_OP(ConvolutionBackpropData, ov::op::v1)
OPENVINO_OP(Cos, ov::op::v0)
OPENVINO_OP(Cosh, ov::op::v0)
OPENVINO_OP(DeformableConvolution, ov::op::v1)
OPENVINO_OP(DeformablePSROIPooling, ov::op::v1)
OPENVINO_OP(DepthToSpace, ov::op::v0)
OPENVINO_OP(DetectionOutput, ov::op::v0)
OPENVINO_OP(Divide, ov::op::v1)
OPENVINO_OP(Elu, ov::op::v0)
OPENVINO_OP(Erf, ov::op::v0)
OPENVINO_OP(Equal, ov::op::v1)
OPENVINO_OP(Exp, ov::op::v0)
OPENVINO_OP(FakeQuantize, ov::op::v0)
OPENVINO_OP(Floor, ov::op::v0)
OPENVINO_OP(FloorMod, ov::op::v1)
OPENVINO_OP(Gather, ov::op::v1)
OPENVINO_OP(GatherTree, ov::op::v1)
OPENVINO_OP(Greater, ov::op::v1)
OPENVINO_OP(GreaterEqual, ov::op::v1)
OPENVINO_OP(GroupConvolution, ov::op::v1)
OPENVINO_OP(GroupConvolutionBackpropData, ov::op::v1)
OPENVINO_OP(GRN, ov::op::v0)
OPENVINO_OP(HardSigmoid, ov::op::v0)
OPENVINO_OP(Interpolate, ov::op::v0)
OPENVINO_OP(Less, ov::op::v1)
OPENVINO_OP(LessEqual, ov::op::v1)
OPENVINO_OP(Log, ov::op::v0)
OPENVINO_OP(LogicalAnd, ov::op::v1)
OPENVINO_OP(LogicalNot, ov::op::v1)
OPENVINO_OP(LogicalOr, ov::op::v1)
OPENVINO_OP(LogicalXor, ov::op::v1)
OPENVINO_OP(LRN, ov::op::v0)
OPENVINO_OP(LSTMCell, ov::op::v0)
OPENVINO_OP(LSTMSequence, ov::op::v0)
OPENVINO_OP(MatMul, ov::op::v0)
OPENVINO_OP(MaxPool, ov::op::v1)
OPENVINO_OP(Maximum, ov::op::v1)
OPENVINO_OP(Minimum, ov::op::v1)
OPENVINO_OP(Mod, ov::op::v1)
OPENVINO_OP(Multiply, ov::op::v1)

OPENVINO_OP(MVN, ov::op::v0)  // Missing in opset1

OPENVINO_OP(Negative, ov::op::v0)
OPENVINO_OP(NonMaxSuppression, ov::op::v1)
OPENVINO_OP(NormalizeL2, ov::op::v0)
OPENVINO_OP(NotEqual, ov::op::v1)
OPENVINO_OP(OneHot, ov::op::v1)
OPENVINO_OP(PRelu, ov::op::v0)
OPENVINO_OP(PSROIPooling, ov::op::v0)
OPENVINO_OP(Pad, ov::op::v1)
OPENVINO_OP(Parameter, ov::op::v0)
OPENVINO_OP(Power, ov::op::v1)
OPENVINO_OP(PriorBox, ov::op::v0)
OPENVINO_OP(PriorBoxClustered, ov::op::v0)
OPENVINO_OP(Proposal, ov::op::v0)
OPENVINO_OP(Range, ov::op::v0)
OPENVINO_OP(Relu, ov::op::v0)
OPENVINO_OP(ReduceMax, ov::op::v1)
OPENVINO_OP(ReduceLogicalAnd, ov::op::v1)
OPENVINO_OP(ReduceLogicalOr, ov::op::v1)
OPENVINO_OP(ReduceMean, ov::op::v1)
OPENVINO_OP(ReduceMin, ov::op::v1)
OPENVINO_OP(ReduceProd, ov::op::v1)
OPENVINO_OP(ReduceSum, ov::op::v1)
OPENVINO_OP(RegionYolo, ov::op::v0)

OPENVINO_OP(ReorgYolo, ov::op::v0)  // Missing in opset1

OPENVINO_OP(Reshape, ov::op::v1)
OPENVINO_OP(Result, ov::op::v0)

// Moved out of opset2, it was added to opset1 by mistake
// OPENVINO_OP(Reverse, ov::op::v1)

OPENVINO_OP(ReverseSequence, ov::op::v0)

// Moved out of opset2, it was added to opset1 by mistake
// OPENVINO_OP(RNNCell, ov::op::v0)

OPENVINO_OP(ROIPooling, ov::op::v0)  // Missing in opset1

OPENVINO_OP(Select, ov::op::v1)
OPENVINO_OP(Selu, ov::op::v0)
OPENVINO_OP(ShapeOf, ov::op::v0)

// Moved out of opset2, it was added to opset1 by mistake
// OPENVINO_OP(ShuffleChannels, ov::op::v0)

OPENVINO_OP(Sign, ov::op::v0)
OPENVINO_OP(Sigmoid, ov::op::v0)
OPENVINO_OP(Sin, ov::op::v0)
OPENVINO_OP(Sinh, ov::op::v0)
OPENVINO_OP(Softmax, ov::op::v1)
OPENVINO_OP(Sqrt, ov::op::v0)
OPENVINO_OP(SpaceToDepth, ov::op::v0)
OPENVINO_OP(Split, ov::op::v1)
OPENVINO_OP(SquaredDifference, ov::op::v0)
OPENVINO_OP(Squeeze, ov::op::v0)
OPENVINO_OP(StridedSlice, ov::op::v1)
OPENVINO_OP(Subtract, ov::op::v1)
OPENVINO_OP(Tan, ov::op::v0)
OPENVINO_OP(Tanh, ov::op::v0)
OPENVINO_OP(TensorIterator, ov::op::v0)
OPENVINO_OP(Tile, ov::op::v0)
OPENVINO_OP(TopK, ov::op::v1)
OPENVINO_OP(Transpose, ov::op::v1)
OPENVINO_OP(Unsqueeze, ov::op::v0)
OPENVINO_OP(VariadicSplit, ov::op::v1)

// Moved out of opset2, it was added to opset1 by mistake
// OPENVINO_OP(Xor, ov::op::v0)

// New operations added in opset2
OPENVINO_OP(Gelu, ov::op::v0)
OPENVINO_OP(BatchToSpace, ov::op::v1)
OPENVINO_OP(SpaceToBatch, ov::op::v1)
