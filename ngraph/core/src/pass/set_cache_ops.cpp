// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/op/util/sub_graph_base.hpp>
#include "ngraph/pass/set_cache_ops.hpp"
#include "itt.hpp"
#include "ngraph/graph_util.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::SetCacheOps, "ngraph::pass::SetCacheOps", 0);

bool pass::SetCacheOps::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    f->reset_cached_ops();
    f->set_cache_ops(m_cache_ops);
    for (auto && node : f->get_ordered_ops())
    {
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node))
        {
            if (auto sub_graph = sub_graph_node->get_function())
            {
                sub_graph->reset_cached_ops();
                sub_graph->set_cache_ops(m_cache_ops);
                run_on_function(sub_graph);
            }
        }
    }
    return false;
}

