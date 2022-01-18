// Copyright (C) 2018-2022 Intel Corporation
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

class OptimizeSequenceTransposes : public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    OptimizeSequenceTransposes();
};

}  // namespace MKLDNNPlugin
