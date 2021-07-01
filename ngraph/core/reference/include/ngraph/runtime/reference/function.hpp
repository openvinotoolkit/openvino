// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <vector>
#include "ngraph/function.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void function(const std::shared_ptr<Function>& function,
                          const HostTensorVector& inputs,
                          HostTensorVector& outputs);
        }
    } // namespace runtime
} // namespace ngraph
