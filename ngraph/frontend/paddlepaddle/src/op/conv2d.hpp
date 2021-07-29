// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "node_context.hpp"

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs conv2d(const NodeContext& node_context);

            }
        } // namespace pdpd
    }     // namespace frontend
} // namespace ngraph
