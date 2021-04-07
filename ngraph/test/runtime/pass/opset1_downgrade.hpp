// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "backend_visibility.hpp"
#include "ngraph/pass/pass.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

namespace ngraph
{
    namespace pass
    {
        class BACKEND_API Opset1Downgrade : public NodePass
        {
        public:
            ///
            /// \brief    Constructor for the Opv1 downgrade transformation pass.
            ///
            /// \details  This transformation pass iterates over all nodes in a graph
            /// and updates version 3 ops to their version 1 equivalents.
            /// All ops in the final graph have op version 1.
            Opset1Downgrade() = default;
            bool run_on_node(std::shared_ptr<ngraph::Node> node) override;
        };
    }
}

NGRAPH_SUPPRESS_DEPRECATED_END
