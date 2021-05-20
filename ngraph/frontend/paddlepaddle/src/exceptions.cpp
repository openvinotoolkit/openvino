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
                ss << "While validating node '" << node.op_type() << '\'';
                return ss.str();
            }

            std::string OpConversionFailurePDPD::get_error_msg_prefix_pdpd(const Node* node)
            {
                std::stringstream ss;
                ss << "While converting node '" << *node << "' with friendly_name '"
                   << node->get_friendly_name() << '\'';
                return ss.str();
            }

        } // namespace pdpd
    }     // namespace frontend
} // namespace ngraph
