// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <ngraph/node.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "rt_info/attribute_parameters.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API BaseMatcherPass;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

class LP_TRANSFORMATIONS_API ngraph::pass::low_precision::BaseMatcherPass : public ngraph::pass::MatcherPass {
public:
    BaseMatcherPass(const AttributeParameters& params = AttributeParameters());
    AttributeParameters params;
};
