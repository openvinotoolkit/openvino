// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <ngraph/opsets/opset7.hpp>

#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void einsum(const HostTensorVector& outputs,
                        const HostTensorVector& inputs,
                        const std::string& equation);
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
