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
#include "ngraph_ops/type_relaxed.hpp"

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

static std::vector<float> simplifyToScale(const std::shared_ptr<ov::opset8::FakeQuantize>& fq_node) {
    auto levels = fq_node->get_levels();
    auto input_low = ov::as_type_ptr<ov::opset8::Constant>(fq_node->get_input_node_shared_ptr(1))->cast_vector<float>();
    auto input_high = ov::as_type_ptr<ov::opset8::Constant>(fq_node->get_input_node_shared_ptr(2))->cast_vector<float>();
    auto output_low = ov::as_type_ptr<ov::opset8::Constant>(fq_node->get_input_node_shared_ptr(3))->cast_vector<float>();
    auto output_high = ov::as_type_ptr<ov::opset8::Constant>(fq_node->get_input_node_shared_ptr(4))->cast_vector<float>();

    std::vector<float> cl, ch, isc, ish, osc, osh;
    for (int i = 0; i < input_low.size(); i++) {
        cl.push_back(input_low[i]);
    }
    for (int i = 0; i < input_high.size(); i++) {
        ch.push_back(input_high[i]);
    }

    for (int i = 0; i < std::max(input_low.size(), input_high.size()); i++) {
        float il = input_low[input_low.size() == 1 ? 0 : i];
        float ih = input_high[input_high.size() == 1 ? 0 : i];

        isc.push_back((levels - 1) / (ih - il));
        ish.push_back(-il * (levels - 1) / (ih - il));
    }

    for (int i = 0; i < std::max(output_low.size(), output_high.size()); i++) {
        float ol = output_low[output_low.size() == 1 ? 0 : i];
        float oh = output_high[output_high.size() == 1 ? 0 : i];

        osc.push_back((oh - ol) / (levels - 1));
        osh.push_back(ol);
    }

    std::vector<float> outScale;

    if (fq_node->get_output_element_type(0) == ngraph::element::u8 &&
            std::all_of(cl.cbegin(), cl.cend(), [](float val) { return val == 0.0f; }) &&
            std::all_of(ish.cbegin(), ish.cend(), [](float val) { return val == 0.0f; }) &&
            std::all_of(osc.cbegin(), osc.cend(), [](float val) { return val == 1.0f; }) &&
            std::all_of(osh.cbegin(), osh.cend(), [](float val) { return val == 0.0f; })) {
        outScale = isc;
    }

    if (fq_node->get_output_element_type(0) == ngraph::element::i8 &&
            std::all_of(ish.cbegin(), ish.cend(), [](float val) { return std::abs(val - 128.f) < 0.001f; }) &&
            std::all_of(osc.cbegin(), osc.cend(), [](float val) { return val == 1.f; }) &&
            std::all_of(osh.cbegin(), osh.cend(), [](float val) { return std::abs(val + 128.f) < 0.001f; })) {
        bool isCropAligned = true;
        for (int i = 0; i < std::max(cl.size(), isc.size()); i++) {
            if (std::abs(cl[cl.size() == 1 ? 0 : i] * isc[isc.size() == 1 ? 0 : i] + 128.f) > 0.001f) {
                isCropAligned = false;
            }
        }

        for (int i = 0; i < std::max(ch.size(), isc.size()); i++) {
            if (std::abs(ch[ch.size() == 1 ? 0 : i] * isc[isc.size() == 1 ? 0 : i] - 127.f) > 0.001f) {
                isCropAligned = false;
            }
        }

        if (isCropAligned) {
            outScale = isc;
        }
    }

    return outScale;
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
    auto fq_m = wrap_type<ov::opset8::FakeQuantize>({inter_m, wrap_type<ov::opset1::Constant>(),
        wrap_type<ov::opset1::Constant>(),
        wrap_type<ov::opset1::Constant>(),
        wrap_type<ov::opset1::Constant>()});
    matcher_pass_callback callback = [=](Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto fq_node = ov::as_type_ptr<ov::opset8::FakeQuantize>(pattern_to_output.at(fq_m).get_node_shared_ptr());
        std::vector<float> fq_scale;
        if (fq_node) {
            fq_scale = simplifyToScale(fq_node);
            if (fq_scale.empty())
                return false;
        }
        bool success = ov::replace_output_update_name(fq_node->output(0), fq_node->input_value(0));
        if (!success) {
            return false;
        }
        auto inter_node = ov::as_type_ptr<InteractionNode>(pattern_to_output.at(inter_m).get_node_shared_ptr());
        inter_node->set_fq_scales(fq_scale);
        inter_node->set_fq_output_type(fq_node->get_output_element_type(0));

        auto replacement = std::make_shared<ngraph::op::TypeRelaxed<InteractionNode>>(*inter_node, fq_node->get_output_element_type(0));
        copy_runtime_info(inter_node, replacement);
        replace_node(inter_node, replacement);
        return success;
    };

    auto m = std::make_shared<Matcher>(fq_m, matcher_name);
    this->register_matcher(m, callback);
}
