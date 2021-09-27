// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace MKLDNNPlugin {

class ReshapePRelu: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ReshapePRelu();
};

}  // namespace MKLDNNPlugin