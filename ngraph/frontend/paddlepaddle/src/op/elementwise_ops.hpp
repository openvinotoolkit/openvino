// Copyright (C) 2018-2021 Intel Corporation
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
                NamedOutputs elementwise_add(const NodeContext& node_context);
                NamedOutputs elementwise_sub(const NodeContext& node_context);
                NamedOutputs elementwise_mul(const NodeContext& node_context);
                NamedOutputs elementwise_div(const NodeContext& node_context);
                NamedOutputs elementwise_min(const NodeContext& node_context);
                NamedOutputs elementwise_max(const NodeContext& node_context);
                NamedOutputs elementwise_pow(const NodeContext& node_context);
            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
