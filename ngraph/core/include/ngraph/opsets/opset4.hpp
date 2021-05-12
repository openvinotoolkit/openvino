// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ops.hpp"

namespace ngraph
{
    namespace opset4
    {
#define NGRAPH_OP(a, b) using b::a;
#include "ngraph/opsets/opset4_tbl.hpp"
#undef NGRAPH_OP
    } // namespace opset4
} // namespace ngraph
