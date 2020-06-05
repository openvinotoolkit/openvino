// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ngraph/ngraph.hpp>
#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API CollapseSubgraph: public ngraph::pass::GraphRewrite {
public:
    CollapseSubgraph();
};

}  // namespace pass
}  // namespace ngraph
