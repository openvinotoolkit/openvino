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

#define ID_SUFFIX(NAME) NAME##_v1
NGRAPH_OP(LessEqual, op::v1)
NGRAPH_OP(LogicalAnd, op::v1)
NGRAPH_OP(LogicalOr, op::v1)
NGRAPH_OP(LogicalXor, op::v1)
NGRAPH_OP(LogicalNot, op::v1)
#undef ID_SUFFIX

#define ID_SUFFIX(NAME) NAME##_v3
NGRAPH_OP(EmbeddingBagOffsetsSum, op::v3)
NGRAPH_OP(EmbeddingBagPackedSum, op::v3)
NGRAPH_OP(EmbeddingSegmentsSum, op::v3)
NGRAPH_OP(ExtractImagePatches, op::v3)
NGRAPH_OP(ShapeOf, op::v3)
NGRAPH_OP(NonZero, op::v3)
#undef ID_SUFFIX
