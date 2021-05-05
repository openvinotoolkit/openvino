// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/runtime/host_tensor.hpp"

namespace ngraph
{
    namespace eval
    {
        AxisSet extract_reduction_axes(const HostTensorPtr& axes, const char* op_name);
    }
} // namespace ngraph
