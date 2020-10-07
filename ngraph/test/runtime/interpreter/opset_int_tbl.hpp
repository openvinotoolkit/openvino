//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#define ID_SUFFIX(NAME) NAME
#include "opset0_tbl.hpp"
#undef ID_SUFFIX

#define ID_SUFFIX(NAME) NAME##_v0
NGRAPH_OP(CTCGreedyDecoder, ngraph::op::v0)
NGRAPH_OP(DetectionOutput, op::v0)
NGRAPH_OP(RNNCell, op::v0)
#undef ID_SUFFIX

#define ID_SUFFIX(NAME) NAME##_v1
NGRAPH_OP(LessEqual, op::v1)
NGRAPH_OP(LogicalAnd, op::v1)
NGRAPH_OP(LogicalOr, op::v1)
NGRAPH_OP(LogicalXor, op::v1)
NGRAPH_OP(LogicalNot, op::v1)
NGRAPH_OP(GatherTree, op::v1)
#undef ID_SUFFIX

#define ID_SUFFIX(NAME) NAME##_v3
NGRAPH_OP(GRUCell, op::v3)
NGRAPH_OP(EmbeddingBagOffsetsSum, op::v3)
NGRAPH_OP(EmbeddingBagPackedSum, op::v3)
NGRAPH_OP(EmbeddingSegmentsSum, op::v3)
NGRAPH_OP(ExtractImagePatches, op::v3)
NGRAPH_OP(ShapeOf, op::v3)
NGRAPH_OP(NonZero, op::v3)
NGRAPH_OP(ScatterNDUpdate, op::v3)
NGRAPH_OP(ScatterUpdate, op::v3)
#undef ID_SUFFIX

#define ID_SUFFIX(NAME) NAME##_v4
NGRAPH_OP(CTCLoss, op::v4)
NGRAPH_OP(LSTMCell, op::v4)
#undef ID_SUFFIX

#define ID_SUFFIX(NAME) NAME##_v5
NGRAPH_OP(LSTMSequence, op::v5)
NGRAPH_OP(GRUSequence, op::v5)
NGRAPH_OP(RNNSequence, op::v5)
#undef ID_SUFFIX
