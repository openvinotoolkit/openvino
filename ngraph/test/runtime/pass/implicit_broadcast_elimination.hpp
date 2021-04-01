// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "backend_visibility.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/pass.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

namespace ngraph
{
    namespace pass
    {
        NodeVector explicit_broadcast(std::shared_ptr<Node>& node);
        class ImplicitBroadcastElimination;
    }
}

class BACKEND_API ngraph::pass::ImplicitBroadcastElimination : public ngraph::pass::NodePass
{
public:
    bool run_on_node(std::shared_ptr<ngraph::Node> node) override;
};

NGRAPH_SUPPRESS_DEPRECATED_END
