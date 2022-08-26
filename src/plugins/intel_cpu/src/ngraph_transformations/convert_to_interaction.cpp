// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_to_interaction.hpp"
#include "op/interaction.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"

ov::intel_cpu::ConvertToInteraction::ConvertToInteraction() {
    MATCHER_SCOPE(ConvertToInteraction);
    using namespace ngraph::pattern;
    auto dense_feature_m = any_input(has_static_rank());
    std::vector<std::shared_ptr<Node>> features_m{dense_feature_m};
    OutputVector features_output{dense_feature_m->output(0)};
    const int sparse_feature_num = 26;
    for (size_t i = 0; i < sparse_feature_num; i++) {
        auto feature = any_input(has_static_rank());
        features_m.push_back(feature);
        features_output.push_back(feature->output(0));
    }
    auto concat_m = wrap_type<ngraph::opset8::Concat>(features_output);
    auto reshape_m = wrap_type<ngraph::opset8::Reshape>({concat_m->output(0), any_input()->output(0)});
    // This transpose is moved due to TransposeMatmul Transformation
    // auto transpose_m = wrap_type<ngraph::opset8::Transpose>({reshape_m->output(0), any_input()->output(0)});
    auto matmul_m = wrap_type<ngraph::opset1::MatMul>({reshape_m, reshape_m});
    auto transpose2_m = wrap_type<ngraph::opset1::Transpose>({matmul_m->output(0), any_input()->output(0)});
    auto reshape2_m = wrap_type<ngraph::opset1::Reshape>({transpose2_m->output(0), any_input()->output(0)});
    auto gather_m = wrap_type<ngraph::opset8::Gather>({reshape2_m->output(0), any_input()->output(0), any_input()->output(0)});
    // This reshape is moved to to EliminateReshape
    // auto reshape3_m = wrap_type<ngraph::opset1::Reshape>({gather_m->output(0), any_input()->output(0)});
    auto transpose3_m = wrap_type<ngraph::opset1::Transpose>({gather_m->output(0), any_input()->output(0)});
    // This reshape is moved to to EliminateReshape
    // auto reshape4_m = wrap_type<ngraph::opset1::Reshape>({transpose3_m->output(0), any_input()->output(0)});
    auto final_concat_m = wrap_type<ngraph::opset1::Concat>({dense_feature_m->output(0), transpose3_m->output(0)});

    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto concat_node = pattern_map.at(concat_m).get_node_shared_ptr();
        auto dense_feature_node = concat_node->input_value(0).get_node_shared_ptr();
        auto final_concat_node = pattern_map.at(final_concat_m).get_node_shared_ptr();
        std::vector<std::shared_ptr<Node>> features_node;

        for (size_t i = 0; i < features_m.size(); i++) {
            auto old_feature_node = pattern_map.at(features_m[i]).get_node_shared_ptr();
            features_node.push_back(old_feature_node);
            //disconnect original consumers of features.
            for (auto& input : old_feature_node->output(0).get_target_inputs()) {
                old_feature_node->output(0).remove_target_input(input);
            }
        }
        auto interaction_node = std::make_shared<InteractionNode>(features_node);
        replace_node(final_concat_node, interaction_node);

        return true;
    };

    auto m = std::make_shared<Matcher>(final_concat_m, matcher_name);
    this->register_matcher(m, callback);
}
