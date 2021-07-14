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
                NamedOutputs rnn(const NodeContext& node);
            }
        } // namespace pdpd
    }     // namespace frontend
} // namespace ngraph
