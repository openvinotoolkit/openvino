// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_to_interaction.hpp"

#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "openvino/opsets/opset1.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "simplify_fakequantize.hpp"
#include "transformations/cpu_opset/x64/op/interaction.hpp"

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
    auto gather_m =
        wrap_type<ov::opset8::Gather>({reshape2_m->output(0), any_input()->output(0), any_input()->output(0)});
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
        for (const auto& i : features_m) {
            auto old_feature_node = pattern_map.at(i).get_node_shared_ptr();
            auto this_feature_shape = old_feature_node->get_output_partial_shape(0);
            // check whether inputs are all equal
            if (!first_feature_shape.compatible(this_feature_shape)) {
                return false;
            }
            first_feature_shape = this_feature_shape;
            features_node.push_back(old_feature_node);
            // disconnect original consumers of features.
            for (auto& input : old_feature_node->output(0).get_target_inputs()) {
                old_feature_node->output(0).remove_target_input(input);
            }
        }
        auto interaction_node = std::make_shared<InteractionNode>(features_node);
        interaction_node->set_friendly_name(final_concat_node->get_friendly_name());
        replace_node(final_concat_node, interaction_node);

        return true;
    };

    auto m = std::make_shared<Matcher>(
        std::make_shared<ov::pass::pattern::op::Or>(OutputVector{final_concat_m1, final_concat_m2}),
        matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::FuseFQtoInteraction::FuseFQtoInteraction() {
    MATCHER_SCOPE(FuseFQtoInteraction);
    using namespace ov::pass::pattern;
    const int input_features = 27;
    OutputVector features_output;
    for (size_t i = 0; i < input_features; i++) {
        auto feature = any_input(has_static_shape());
        features_output.push_back(feature->output(0));
    }
    auto inter_m = wrap_type<InteractionNode>(features_output);
    auto fq_m = wrap_type<ov::opset8::FakeQuantize>({inter_m,
                                                     wrap_type<ov::opset1::Constant>(),
                                                     wrap_type<ov::opset1::Constant>(),
                                                     wrap_type<ov::opset1::Constant>(),
                                                     wrap_type<ov::opset1::Constant>()});
    matcher_pass_callback callback = [=](Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto fq_node = ov::as_type_ptr<ov::opset8::FakeQuantize>(pattern_to_output.at(fq_m).get_node_shared_ptr());
        std::vector<float> fq_scale;
        if (fq_node) {
            fq_scale = simplifyToScale(fq_node, 0.001f);
            if (fq_scale.empty()) {
                return false;
            }
        }
        bool success = ov::replace_output_update_name(fq_node->output(0), fq_node->input_value(0));
        if (!success) {
            return false;
        }
        auto inter_node = ov::as_type_ptr<InteractionNode>(pattern_to_output.at(inter_m).get_node_shared_ptr());
        inter_node->set_fq_scales(fq_scale);
        inter_node->set_output_type(0, fq_node->get_output_element_type(0), inter_node->get_output_partial_shape(0));

        auto replacement =
            std::make_shared<ov::op::TypeRelaxed<InteractionNode>>(*inter_node, fq_node->get_output_element_type(0));
        copy_runtime_info(inter_node, replacement);
        replace_node(inter_node, replacement);
        return success;
    };

    auto m = std::make_shared<Matcher>(fq_m, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::ConvertInteractionInt8::ConvertInteractionInt8() {
    MATCHER_SCOPE(ConvertInteractionInt8);
    using namespace ov::pass::pattern;
    const int sparse_features = 26;
    OutputVector features_output;
    auto dense_fq_m = wrap_type<ov::opset8::FakeQuantize>({any_input(has_static_shape()),
                                                           wrap_type<ov::opset1::Constant>(),
                                                           wrap_type<ov::opset1::Constant>(),
                                                           wrap_type<ov::opset1::Constant>(),
                                                           wrap_type<ov::opset1::Constant>()});
    std::vector<std::shared_ptr<Node>> features_m{dense_fq_m};
    features_output.push_back(dense_fq_m->output(0));
    for (size_t i = 0; i < sparse_features; i++) {
        auto feature = any_input(has_static_shape());
        features_m.push_back(feature);
        features_output.push_back(feature->output(0));
    }

    auto concat_m = wrap_type<ov::opset8::Concat>(features_output);
    auto reshape_m = wrap_type<ov::opset8::Reshape>({concat_m->output(0), any_input()->output(0)});
    auto matmul_m = wrap_type<ov::opset1::MatMul>({reshape_m, reshape_m});
    auto sparse_fq = wrap_type<ov::opset8::FakeQuantize>({matmul_m->output(0),
                                                          wrap_type<ov::opset1::Constant>(),
                                                          wrap_type<ov::opset1::Constant>(),
                                                          wrap_type<ov::opset1::Constant>(),
                                                          wrap_type<ov::opset1::Constant>()});
    auto transpose2_m = wrap_type<ov::opset1::Transpose>({sparse_fq->output(0), any_input()->output(0)});
    auto reshape2_m = wrap_type<ov::opset1::Reshape>({transpose2_m->output(0), any_input()->output(0)});
    auto gather_m =
        wrap_type<ov::opset8::Gather>({reshape2_m->output(0), any_input()->output(0), any_input()->output(0)});
    auto transpose3_m = wrap_type<ov::opset1::Transpose>({gather_m->output(0), any_input()->output(0)});
    auto final_concat_m = wrap_type<ov::opset1::Concat>({dense_fq_m->output(0), transpose3_m->output(0)});

    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto concat_node = pattern_map.at(concat_m).get_node_shared_ptr();
        auto dense_fq_node = pattern_map.at(dense_fq_m).get_node_shared_ptr();
        auto sparse_fq_node = pattern_map.at(sparse_fq).get_node_shared_ptr();
        auto final_concat_node = pattern_map.at(final_concat_m).get_node_shared_ptr();
        std::vector<std::shared_ptr<Node>> features_node;
        auto first_feature_shape = dense_fq_node->get_output_partial_shape(0);
        for (const auto& i : features_m) {
            auto old_feature_node = pattern_map.at(i).get_node_shared_ptr();
            auto this_feature_shape = old_feature_node->get_output_partial_shape(0);
            // check whether inputs are all equal
            if (!first_feature_shape.compatible(this_feature_shape)) {
                return false;
            }
            first_feature_shape = this_feature_shape;
            features_node.push_back(old_feature_node);
            // disconnect original consumers of features.
            for (auto& input : old_feature_node->output(0).get_target_inputs()) {
                old_feature_node->output(0).remove_target_input(input);
            }
        }
        // check whether the inputs to concat have same fakequantize parameters
        for (size_t i = 1; i < dense_fq_node->get_input_size(); i++) {
            auto dense_const = ov::as_type_ptr<ov::opset8::Constant>(dense_fq_node->get_input_node_shared_ptr(i))
                                   ->cast_vector<float>();
            auto sparse_const = ov::as_type_ptr<ov::opset8::Constant>(sparse_fq_node->get_input_node_shared_ptr(i))
                                    ->cast_vector<float>();
            if (dense_const.size() != sparse_const.size() || dense_const != sparse_const) {
                return false;
            }
        }

        auto interaction_node = std::make_shared<InteractionNode>(features_node);
        interaction_node->set_friendly_name(final_concat_node->get_friendly_name());
        auto fq_inter_node = dense_fq_node->clone_with_new_inputs({interaction_node,
                                                                   dense_fq_node->get_input_node_shared_ptr(1),
                                                                   dense_fq_node->get_input_node_shared_ptr(2),
                                                                   dense_fq_node->get_input_node_shared_ptr(3),
                                                                   dense_fq_node->get_input_node_shared_ptr(4)});
        fq_inter_node->set_friendly_name(final_concat_node->get_friendly_name());
        replace_node(final_concat_node, fq_inter_node);
        return true;
    };

    auto m = std::make_shared<Matcher>(final_concat_m, matcher_name);
    this->register_matcher(m, callback);
}
