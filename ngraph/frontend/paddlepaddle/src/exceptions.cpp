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
                OpValidationFailurePDPD::get_error_msg_prefix_pdpd(const pdpd::NodeContext& node)
            {
                std::stringstream ss;
                ss << "While validating node '" << node.get_op_type() << '\'';
                return ss.str();
            }
        } // namespace pdpd
    }     // namespace frontend
} // namespace ngraph
