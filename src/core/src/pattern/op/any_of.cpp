// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/any_of.hpp"

#include "openvino/pass/pattern/matcher.hpp"

bool ov::pass::pattern::op::AnyOf::match_value(Matcher* matcher,
                                               const Output<Node>& pattern_value,
                                               const Output<Node>& graph_value) {
    matcher->add_node(graph_value);
    return m_predicate(matcher->get_symbols(), graph_value) && ([&]() {
               for (const auto& arg : graph_value.get_node_shared_ptr()->input_values()) {
                   auto saved = matcher->start_match();
                   if (matcher->match_value(input_value(0), arg)) {
                       return saved.finish(true);
                   }
               }
               return false;
           }());
}
