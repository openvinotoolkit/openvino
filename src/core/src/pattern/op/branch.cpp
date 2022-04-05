// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pattern/op/branch.hpp"

#include "ngraph/pattern/matcher.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(pattern::op::Branch);

bool pattern::op::Branch::match_value(Matcher* matcher,
                                      const Output<Node>& pattern_value,
                                      const Output<Node>& graph_value) {
    return matcher->match_value(get_destination(), graph_value);
}
