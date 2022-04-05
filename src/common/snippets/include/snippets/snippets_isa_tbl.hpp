// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef NGRAPH_OP
#warning "NGRAPH_OP not defined"
#define NGRAPH_OP(x, y)
#endif

// SnippetS dialect
NGRAPH_OP(Load, ngraph::snippets::op)
NGRAPH_OP(ScalarLoad, ngraph::snippets::op)
NGRAPH_OP(VectorLoad, ngraph::snippets::op)
NGRAPH_OP(BlockedLoad, ngraph::snippets::op)
NGRAPH_OP(BroadcastLoad, ngraph::snippets::op)

NGRAPH_OP(Store, ngraph::snippets::op)
NGRAPH_OP(ScalarStore, ngraph::snippets::op)
NGRAPH_OP(VectorStore, ngraph::snippets::op)

NGRAPH_OP(BroadcastMove, ngraph::snippets::op)
NGRAPH_OP(Scalar, ngraph::snippets::op)
NGRAPH_OP(Nop, ngraph::snippets::op)

// Layout-oblivious from opset1

// opset completeness
NGRAPH_OP(Constant, ngraph::op)
NGRAPH_OP(Parameter, ngraph::op::v0)
NGRAPH_OP(BlockedParameter, ngraph::snippets::op)
NGRAPH_OP(Result, ngraph::op::v0)
NGRAPH_OP(Broadcast, ngraph::op::v1)

// unary
NGRAPH_OP(Abs, ngraph::op::v0)
NGRAPH_OP(Acos, ngraph::op::v0)
NGRAPH_OP(Asin, ngraph::op::v0)
NGRAPH_OP(Atan, ngraph::op::v0)
NGRAPH_OP(Ceiling, ngraph::op::v0)
NGRAPH_OP(Clamp, ngraph::op::v0)
NGRAPH_OP(Cos, ngraph::op::v0)
NGRAPH_OP(Cosh, ngraph::op::v0)
NGRAPH_OP(Elu, ngraph::op::v0)
NGRAPH_OP(Erf, ngraph::op::v0)
NGRAPH_OP(Exp, ngraph::op::v0)
NGRAPH_OP(Floor, ngraph::op::v0)
NGRAPH_OP(HardSigmoid, ngraph::op::v0)
NGRAPH_OP(Log, ngraph::op::v0)
NGRAPH_OP(LogicalNot, ngraph::op::v1)
NGRAPH_OP(Negative, ngraph::op::v0)
NGRAPH_OP(Relu, ngraph::op::v0)
NGRAPH_OP(Round, ngraph::op::v5)
NGRAPH_OP(Selu, ngraph::op::v0)
NGRAPH_OP(Sign, ngraph::op::v0)
NGRAPH_OP(Sigmoid, ngraph::op::v0)
NGRAPH_OP(Sin, ngraph::op::v0)
NGRAPH_OP(Sinh, ngraph::op::v0)
NGRAPH_OP(Sqrt, ngraph::op::v0)
NGRAPH_OP(Tan, ngraph::op::v0)
NGRAPH_OP(Tanh, ngraph::op::v0)

// binary
NGRAPH_OP(Add, ngraph::op::v1)
NGRAPH_OP(Divide, ngraph::op::v1)
NGRAPH_OP(Equal, ngraph::op::v1)
NGRAPH_OP(FloorMod, ngraph::op::v1)
NGRAPH_OP(Greater, ngraph::op::v1)
NGRAPH_OP(GreaterEqual, ngraph::op::v1)
NGRAPH_OP(Less, ngraph::op::v1)
NGRAPH_OP(LessEqual, ngraph::op::v1)
NGRAPH_OP(LogicalAnd, ngraph::op::v1)
NGRAPH_OP(LogicalOr, ngraph::op::v1)
NGRAPH_OP(LogicalXor, ngraph::op::v1)
NGRAPH_OP(Maximum, ngraph::op::v1)
NGRAPH_OP(Minimum, ngraph::op::v1)
NGRAPH_OP(Mod, ngraph::op::v1)
NGRAPH_OP(Multiply, ngraph::op::v1)
NGRAPH_OP(NotEqual, ngraph::op::v1)
NGRAPH_OP(Power, ngraph::op::v1)
NGRAPH_OP(PRelu, ngraph::op::v0)
NGRAPH_OP(SquaredDifference, ngraph::op::v0)
NGRAPH_OP(Subtract, ngraph::op::v1)
NGRAPH_OP(Xor, ngraph::op::v0)
