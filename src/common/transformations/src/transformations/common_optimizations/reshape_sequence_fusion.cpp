// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/reshape_sequence_fusion.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"


NGRAPH_RTTI_DEFINITION(ngraph::pass::ReshapeSequenceFusion, "ReshapeSequenceFusion", 0);

namespace {
bool has_valid_pattern(const std::shared_ptr<ngraph::Node> & node) {
    auto const_node = std::dynamic_pointer_cast<ngraph::opset8::Constant>(node);
    if (!const_node) return false;
    const auto & values = const_node->cast_vector<int64_t>();
    // We can not fuse Reshapes if their pattern values have special numbers like -1 and 0
    return std::all_of(values.cbegin(), values.cend(), [](int64_t value) { return value > 0;});
}
}

ngraph::pass::ReshapeSequenceFusion::ReshapeSequenceFusion() {
    MATCHER_SCOPE(ReshapeSequenceFusion);
    auto reshape_input = pattern::any_input();
    auto reshape_a_pattern = pattern::wrap_type<opset8::Constant>();
    auto reshape_a = pattern::wrap_type<opset8::Reshape>({reshape_input, reshape_a_pattern}, pattern::consumers_count(1));
    auto reshape_b_pattern = pattern::wrap_type<opset8::Constant>();
    auto reshape_b = pattern::wrap_type<opset8::Reshape>({reshape_a, reshape_b_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto & pattern_map = m.get_pattern_value_map();
        auto input = pattern_map.at(reshape_input);
        auto reshape = m.get_match_root();

        auto pattern_a = pattern_map.at(reshape_a_pattern).get_node_shared_ptr();
        auto pattern_b = pattern_map.at(reshape_b_pattern).get_node_shared_ptr();
        // skip reshapes which patterns contain special numbers like -1 or 0
        if (!has_valid_pattern(pattern_a) || !has_valid_pattern(pattern_b)) {
            return false;
        }

        // vector of nodes which runtime info must be copied
        NodeVector nodes{pattern_map.at(reshape_a).get_node_shared_ptr(), reshape};
        while (std::dynamic_pointer_cast<opset8::Reshape>(input.get_node_shared_ptr())) {
            auto node = input.get_node_shared_ptr();
            if (!has_valid_pattern(node->get_input_node_shared_ptr(1)) ||
                input.get_target_inputs().size() != 1) {
                break;
            }
            nodes.push_back(node);
            input = node->input_value(0);
        }

        reshape->input(0).replace_source_output(input);
        copy_runtime_info(nodes, reshape);
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_b, matcher_name);
    this->register_matcher(m, callback);
}
