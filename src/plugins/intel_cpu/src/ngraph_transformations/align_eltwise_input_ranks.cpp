// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "align_eltwise_input_ranks.hpp"
#include "utils/rt_info/memory_formats_attribute.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::AlignEltwiseInputRanks, "AlignEltwiseInputRanks", 0);

using namespace ngraph;

MKLDNNPlugin::AlignEltwiseInputRanks::AlignEltwiseInputRanks() {
    auto input_pattern = pattern::any_input(pattern::has_static_rank());
    auto const_pattern = pattern::wrap_type<opset8::Constant>();
    auto pattern = pattern::wrap_type<opset8::SquaredDifference,
                                      op::util::BinaryElementwiseComparison,
                                      op::util::BinaryElementwiseLogical,
                                      op::util::BinaryElementwiseArithmetic>({input_pattern, const_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();
        std::string input_formats = ngraph::getMKLDNNInputMemoryFormats(node);
        if (input_formats.size() > 0)
            return false;

        const auto& pattern_value_map = m.get_pattern_value_map();
        const auto& input = pattern_value_map.at(input_pattern);
        const auto& constant = pattern_value_map.at(const_pattern);
        const auto& const_shape = constant.get_shape();
        auto rank = static_cast<size_t>(node->get_output_partial_shape(0).rank().get_length());

        if (rank <= const_shape.size() || shape_size(const_shape) == 1)
            return false;

        auto diff = rank - const_shape.size();
        std::vector<int> indexes(diff);
        std::iota(indexes.begin(), indexes.end(), 0);
        auto indexes_const = opset8::Constant::create(element::i32, Shape{diff}, indexes);
        auto unsqueeze = std::make_shared<opset8::Unsqueeze>(constant, indexes_const);

        auto new_node = node->clone_with_new_inputs({input, unsqueeze});
        new_node->set_friendly_name(node->get_friendly_name());
        copy_runtime_info(node, {new_node, unsqueeze});
        replace_node(node, new_node);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(pattern, "AlignEltwiseInputRanks");
    this->register_matcher(m, callback);
}
