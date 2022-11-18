// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gna_data_types.hpp"

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

class GatherIESubstitute : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GatherIESubstitute();
};

class GatherRemove : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GatherRemove(GNAPluginNS::SubgraphCPUMap * subgraph_cpu_map = nullptr) : m_subgraph_cpu_map(subgraph_cpu_map) {}
    bool run_on_model(const std::shared_ptr<ngraph::Function>& f) override;
private:
    GNAPluginNS::SubgraphCPUMap * m_subgraph_cpu_map;
};

} // namespace pass
} // namespace intel_gna
} // namespace ov

