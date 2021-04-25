// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ops.hpp"
#include "ngraph/opsets/opset1.hpp"

namespace ngraph
{
    namespace opset2
    {
#define NGRAPH_OP(a, b) using b::a;
#include "ngraph/opsets/opset2_tbl.hpp"
#undef NGRAPH_OP
    } // namespace opset2
} // namespace ngraph
