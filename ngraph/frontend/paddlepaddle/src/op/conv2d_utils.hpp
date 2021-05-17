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
                ngraph::op::PadType get_auto_pad(const NodeContext& node);
                std::pair<CoordinateDiff, CoordinateDiff> get_pads(const NodeContext& node);
                std::shared_ptr<Node> get_reshaped_filter(const Output<Node>& filters, int32_t groups);

            }
        } // namespace pdpd
    }     // namespace frontend
} // namespace ngraph
