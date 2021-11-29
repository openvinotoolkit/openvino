// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/base_matcher_pass.hpp"
#include <ngraph/node.hpp>
#include "low_precision/rt_info/attribute_parameters.hpp"

using namespace ngraph;
using namespace ngraph::pass::low_precision;

ngraph::pass::low_precision::BaseMatcherPass::BaseMatcherPass(const AttributeParameters& params) : params(params) {
}
