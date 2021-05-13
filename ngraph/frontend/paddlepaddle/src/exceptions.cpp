// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paddlepaddle_frontend/exceptions.hpp"
#include "node_context.hpp"

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            std::string
                NodeValidationFailurePDPD::get_error_msg_prefix_pdpd(const pdpd::NodeContext& node)
            {
                return " \nNodeValidationFailure: validation failed for " + node.op_type() +
                       " PaddlePaddle node.";
            }

        } // namespace pdpd
    }     // namespace frontend
} // namespace ngraph
