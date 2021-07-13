// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace MKLDNNPlugin {

class SplitConvConcatPattern: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SplitConvConcatPattern();
};

}  // namespace MKLDNNPlugin
