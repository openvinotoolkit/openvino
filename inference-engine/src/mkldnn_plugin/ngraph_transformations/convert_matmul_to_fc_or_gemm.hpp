// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace MKLDNNPlugin {

class ConvertMatMulToFC: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertMatMulToFC();
};

class ConvertMatMulToGemm: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertMatMulToGemm();
};

}  // namespace MKLDNNPlugin
