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
                // TODO support other interp such as linear trilinear, bicubic. etc #55397
                NamedOutputs nearest_interp_v2(const NodeContext& node_context);
                NamedOutputs bilinear_interp_v2(const NodeContext& node_context);
            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
