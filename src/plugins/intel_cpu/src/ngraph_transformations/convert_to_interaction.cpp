// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_to_interaction.hpp"
#include "op/interaction.hpp"
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"

ov::intel_cpu::ConvertToInteraction::ConvertToInteraction() {
    MATCHER_SCOPE(ConvertToInteraction);
    using namespace ov::pass::pattern;
    auto dense_feature_m = any_input(has_static_shape());
    std::vector<std::shared_ptr<Node>> features_m{dense_feature_m};
    OutputVector features_output{dense_feature_m->output(0)};
    const int sparse_feature_num = 26;
    for (size_t i = 0; i < sparse_feature_num; i++) {
        auto feature = any_input(has_static_shape());
        features_m.push_back(feature);
        features_output.push_back(feature->output(0));
    }
    auto concat_m = wrap_type<ov::opset8::Concat>(features_output);
    auto reshape_m = wrap_type<ov::opset8::Reshape>({concat_m->output(0), any_input()->output(0)});
    auto matmul_m = wrap_type<ov::opset1::MatMul>({reshape_m, reshape_m});
    auto transpose2_m = wrap_type<ov::opset1::Transpose>({matmul_m->output(0), any_input()->output(0)});
    auto reshape2_m = wrap_type<ov::opset1::Reshape>({transpose2_m->output(0), any_input()->output(0)});
    auto gather_m = wrap_type<ov::opset8::Gather>({reshape2_m->output(0), any_input()->output(0), any_input()->output(0)});
    auto transpose3_m = wrap_type<ov::opset1::Transpose>({gather_m->output(0), any_input()->output(0)});
    auto final_concat_m1 = wrap_type<ov::opset1::Concat>({dense_feature_m->output(0), transpose3_m->output(0)});

    auto reshape3_m = wrap_type<ov::opset1::Reshape>({gather_m->output(0), any_input()->output(0)});
    auto transpose4_m = wrap_type<ov::opset1::Transpose>({reshape3_m->output(0), any_input()->output(0)});
    auto reshape4_m = wrap_type<ov::opset1::Reshape>({transpose4_m->output(0), any_input()->output(0)});
    auto final_concat_m2 = wrap_type<ov::opset1::Concat>({dense_feature_m->output(0), reshape4_m->output(0)});

    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto concat_node = pattern_map.at(concat_m).get_node_shared_ptr();
        auto dense_feature_node = concat_node->input_value(0).get_node_shared_ptr();
        std::shared_ptr<Node> final_concat_node;
        if (pattern_map.find(final_concat_m1) != pattern_map.end()) {
            final_concat_node = pattern_map.at(final_concat_m1).get_node_shared_ptr();
        } else if (pattern_map.find(final_concat_m2) != pattern_map.end()) {
            final_concat_node = pattern_map.at(final_concat_m2).get_node_shared_ptr();
        }
        std::vector<std::shared_ptr<Node>> features_node;
        auto first_feature_shape = dense_feature_node->get_output_partial_shape(0);
        for (size_t i = 0; i < features_m.size(); i++) {
            auto old_feature_node = pattern_map.at(features_m[i]).get_node_shared_ptr();
            auto this_feature_shape = old_feature_node->get_output_partial_shape(0);
            //check whether inputs are all equal
            if (!first_feature_shape.compatible(this_feature_shape)) {
                return false;
            }
            first_feature_shape = this_feature_shape;
            features_node.push_back(old_feature_node);
            //disconnect original consumers of features.
            for (auto& input : old_feature_node->output(0).get_target_inputs()) {
                old_feature_node->output(0).remove_target_input(input);
            }
        }
        auto interaction_node = std::make_shared<InteractionNode>(features_node);
        interaction_node->set_friendly_name(final_concat_node->get_friendly_name());
        replace_node(final_concat_node, interaction_node);

        return true;
    };

    auto m = std::make_shared<Matcher>(std::make_shared<ov::pass::pattern::op::Or>(OutputVector{ final_concat_m1, final_concat_m2}), matcher_name);
    this->register_matcher(m, callback);
}
