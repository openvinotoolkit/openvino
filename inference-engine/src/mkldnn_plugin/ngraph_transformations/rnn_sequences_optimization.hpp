// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace MKLDNNPlugin {

class OptimizeGRUSequenceTransposes : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    OptimizeGRUSequenceTransposes();
};

class OptimizeLSTMSequenceTransposes : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    OptimizeLSTMSequenceTransposes();
};

class OptimizeRNNSequenceTransposes : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    OptimizeRNNSequenceTransposes();
};

}  // namespace MKLDNNPlugin
