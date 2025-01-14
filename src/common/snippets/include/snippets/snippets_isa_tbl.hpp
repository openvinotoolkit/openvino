// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef OV_OP
#warning "OV_OP not defined"
#define OV_OP(x, y)
#endif

// SnippetS dialect
OV_OP(Load, ov::snippets::op)
OV_OP(LoadReorder, ov::snippets::op)
OV_OP(LoopBegin, ov::snippets::op)
OV_OP(LoopEnd, ov::snippets::op)
OV_OP(Brgemm, ov::snippets::op)
OV_OP(BroadcastLoad, ov::snippets::op)
OV_OP(Reshape, ov::snippets::op)
OV_OP(Reorder, ov::snippets::op)

OV_OP(Store, ov::snippets::op)

OV_OP(BroadcastMove, ov::snippets::op)
OV_OP(Scalar, ov::snippets::op)
OV_OP(Nop, ov::snippets::op)
OV_OP(RankNormalization, ov::snippets::op)
OV_OP(ReduceMax, ov::snippets::op)
OV_OP(ReduceSum, ov::snippets::op)

#ifdef SNIPPETS_DEBUG_CAPS
OV_OP(PerfCountBegin, ov::snippets::op)
OV_OP(PerfCountEnd, ov::snippets::op)
#endif

// Layout-oblivious from opset1

// opset completeness
OV_OP(Constant, ov::op::v0)
OV_OP(Parameter, ov::op::v0)
OV_OP(Result, ov::op::v0)
OV_OP(Broadcast, ov::op::v1)
OV_OP(ConvertTruncation, ov::snippets::op)
OV_OP(ConvertSaturation, ov::snippets::op)

// unary
OV_OP(Abs, ov::op::v0)
OV_OP(Acos, ov::op::v0)
OV_OP(Asin, ov::op::v0)
OV_OP(Atan, ov::op::v0)
OV_OP(Ceiling, ov::op::v0)
OV_OP(Clamp, ov::op::v0)
OV_OP(Cos, ov::op::v0)
OV_OP(Cosh, ov::op::v0)
OV_OP(Elu, ov::op::v0)
OV_OP(Erf, ov::op::v0)
OV_OP(Exp, ov::op::v0)
OV_OP(Floor, ov::op::v0)
OV_OP(HardSigmoid, ov::op::v0)
OV_OP(Log, ov::op::v0)
OV_OP(LogicalNot, ov::op::v1)
OV_OP(Negative, ov::op::v0)
OV_OP(Relu, ov::op::v0)
OV_OP(Round, ov::op::v5)
OV_OP(Selu, ov::op::v0)
OV_OP(Sign, ov::op::v0)
OV_OP(Sigmoid, ov::op::v0)
OV_OP(Sin, ov::op::v0)
OV_OP(Sinh, ov::op::v0)
OV_OP(Sqrt, ov::op::v0)
OV_OP(Tan, ov::op::v0)
OV_OP(Tanh, ov::op::v0)

// binary
OV_OP(Add, ov::op::v1)
OV_OP(Divide, ov::op::v1)
OV_OP(Equal, ov::op::v1)
OV_OP(FloorMod, ov::op::v1)
OV_OP(Greater, ov::op::v1)
OV_OP(GreaterEqual, ov::op::v1)
OV_OP(Less, ov::op::v1)
OV_OP(LessEqual, ov::op::v1)
OV_OP(LogicalAnd, ov::op::v1)
OV_OP(LogicalOr, ov::op::v1)
OV_OP(LogicalXor, ov::op::v1)
OV_OP(Maximum, ov::op::v1)
OV_OP(Minimum, ov::op::v1)
OV_OP(Mod, ov::op::v1)
OV_OP(Multiply, ov::op::v1)
OV_OP(NotEqual, ov::op::v1)
OV_OP(Power, ov::op::v1)
OV_OP(PRelu, ov::op::v0)
OV_OP(SquaredDifference, ov::op::v0)
OV_OP(Subtract, ov::op::v1)
OV_OP(Xor, ov::op::v0)
