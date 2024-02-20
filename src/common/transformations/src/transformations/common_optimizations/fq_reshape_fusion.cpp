// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fq_reshape_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::FakeQuantizeReshapeFusion::FakeQuantizeReshapeFusion() {
    MATCHER_SCOPE(FakeQuantizeReshapeFusion);
    const auto fq_node_p = ov::pass::pattern::wrap_type<ov::op::v0::FakeQuantize>(
        {ov::pass::pattern::wrap_type<ov::op::v0::Constant>(),  // for weights only
         pattern::any_input(),
         pattern::any_input(),
         pattern::any_input(),
         pattern::any_input()},
        pattern::consumers_count(1));
    const auto reshape_node_p = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>(
        {fq_node_p, pattern::any_input()},
        [](const Output<Node>& output) {
            // WA: check that all Reshape node consumers are not GroupConvolution operations
            const auto& target_inputs = output.get_target_inputs();
            return std::all_of(target_inputs.begin(), target_inputs.end(), [](const Input<Node>& input) {
                return input.get_node()->get_type_info() != ov::op::v1::GroupConvolution::get_type_info_static();
            });
        });

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto fq_node = pattern_map.at(fq_node_p).get_node_shared_ptr();
        if (fq_node->is_dynamic())
            return false;
        const auto& reshape_node = pattern_map.at(reshape_node_p).get_node_shared_ptr();
        const auto& original_data_rank = fq_node->get_input_shape(0).size();
        OutputVector renewed_inputs = {
            reshape_node->clone_with_new_inputs({fq_node->input_value(0), reshape_node->input_value(1)})};
        for (auto i = 1; i < 5; ++i) {
            Output<Node> limit_input = fq_node->input_value(i);
            auto limit_shape = limit_input.get_shape();
            OPENVINO_ASSERT(limit_shape.size() <= original_data_rank, "FakeQuantize limit input has unexpected rank");
            if (limit_shape.size() < original_data_rank)  // aligning limit rank with data rank
                limit_shape.insert(limit_shape.begin(), original_data_rank - limit_shape.size(), uint64_t(1));
            OPENVINO_ASSERT(limit_shape.size() == original_data_rank, "FakeQuantize limit input has unexpected rank");
            const auto& limit_size = shape_size(limit_shape);
            const auto& max_element = *std::max_element(limit_shape.begin(), limit_shape.end());
            if (max_element == limit_size) {  // per-tensor / per-channel limit
                auto new_limit_shape = reshape_node->get_output_shape(0);
                std::transform(new_limit_shape.begin(),
                               new_limit_shape.end(),
                               new_limit_shape.begin(),
                               [max_element](size_t& dim) {
                                   return dim == max_element ? max_element : 1;
                               });
                const auto& new_limit_size = shape_size(new_limit_shape);
                if (new_limit_size == limit_size) {  // we tracked future channel placement
                    if (new_limit_shape == limit_input.get_shape())
                        renewed_inputs.push_back(limit_input);
                    else
                        renewed_inputs.push_back(reshape_node->clone_with_new_inputs(
                            {limit_input,
                             ov::op::v0::Constant::create(element::i64, {new_limit_shape.size()}, new_limit_shape)}));
                    continue;
                }
            }
            // resulting FQ will become or already is more than per-tensor / per-channel
            return false;
        }
        for (auto& new_input : renewed_inputs)
            copy_runtime_info({reshape_node, fq_node}, new_input.get_node_shared_ptr());
        const auto new_fq_node = fq_node->clone_with_new_inputs(renewed_inputs);
        replace_node(reshape_node, new_fq_node);
        new_fq_node->set_friendly_name(reshape_node->get_friendly_name());
        copy_runtime_info({fq_node, reshape_node}, new_fq_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reshape_node_p, matcher_name);
    this->register_matcher(m, callback);
}
