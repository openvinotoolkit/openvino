// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pattern/op/or.hpp"

#include "ngraph/pattern/matcher.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(pattern::op::Or);

bool pattern::op::Or::match_value(Matcher* matcher,
                                  const Output<Node>& pattern_value,
                                  const Output<Node>& graph_value) {
    for (auto input_value : input_values()) {
        auto saved = matcher->start_match();
        if (matcher->match_value(input_value, graph_value)) {
            auto& pattern_map = matcher->get_pattern_value_map();
            pattern_map[input_value.get_node_shared_ptr()] = graph_value;
            return saved.finish(true);
        }
    }
    return false;
}
