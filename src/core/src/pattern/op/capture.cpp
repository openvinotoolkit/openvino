// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pattern/op/capture.hpp"

#include "ngraph/pattern/matcher.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(pattern::op::Capture);

bool pattern::op::Capture::match_value(Matcher* matcher,
                                       const Output<Node>& pattern_value,
                                       const Output<Node>& graph_value) {
    matcher->capture(m_static_nodes);
    return true;
}
