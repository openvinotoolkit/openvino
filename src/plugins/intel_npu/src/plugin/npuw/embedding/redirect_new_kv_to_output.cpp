// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "redirect_new_kv_to_output.hpp"

#include "../logging.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace opp = ov::pass::pattern;

namespace {

// specific function that match subgraph appeared as result of lpt transformations
auto match_down_up_convert_subgraph_after_lpt = [](const ov::Output<ov::Node>& input) {
    auto upconvert = opp::wrap_type<ov::op::v0::Convert>({input}, opp::type_matches(ov::element::f32));

    auto upscale = opp::wrap_type<ov::op::v0::Constant>(opp::rank_equals(0));
    auto upmul = opp::wrap_type<ov::op::v1::Multiply>({upconvert, upscale});

    auto downscale = opp::wrap_type<ov::op::v0::Constant>(opp::rank_equals(0));
    auto downmul = opp::wrap_type<ov::op::v1::Multiply>({upmul, downscale});

    auto downconvert =
        opp::wrap_type<ov::op::v0::Convert>({downmul},
                                            opp::type_matches_any({ov::element::f8e4m3, ov::element::f8e5m2}));

    return downconvert;
};

}  // namespace

class RedirectNewKvToOutputMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::patterns::RedirectNewKvToOutputMatcher");

    RedirectNewKvToOutputMatcher() {
        // example of fp8 inputs to concat
        // input0 : float8e4m3[1,32,1151,96]
        // input1 : float8e4m3[1,32,1,96]

        // parameter, or down_up subgraph case always works for current models-set
        // with both with fp8-optimisation flag enabled and without it
        // TODO: this matcher logic better to cover with unit-tests
        auto input0 = opp::wrap_type<ov::op::v0::Parameter>();
        auto input0_or =
            std::make_shared<opp::op::Or>(ov::OutputVector{input0, match_down_up_convert_subgraph_after_lpt(input0)});

        auto input1 = opp::any_input();

        auto kv_concat = opp::wrap_type<ov::op::v0::Concat>({input0_or, input1});
        auto result1 = opp::wrap_type<ov::op::v0::Result>(kv_concat);
        auto result2 = opp::wrap_type<ov::op::v0::Result>(match_down_up_convert_subgraph_after_lpt(kv_concat));

        auto result_or = std::make_shared<opp::op::Or>(ov::OutputVector{result1, result2});

        ov::matcher_pass_callback callback = [=](opp::Matcher& m) {
            auto& pattern_to_output = m.get_pattern_value_map();
            auto matched_concat = pattern_to_output.at(kv_concat).get_node_shared_ptr();

            auto c0 = matched_concat->input(0).get_source_output();
            auto c1 = matched_concat->input(1).get_source_output();

            LOG_DEBUG(m.get_name() << ": input0.shape=" << c0.get_shape());
            LOG_DEBUG(m.get_name() << ": input1.shape=" << c1.get_shape());
            LOG_DEBUG(m.get_name() << ": concat=" << matched_concat->get_friendly_name());
            LOG_DEBUG(m.get_name() << ": new_kv=" << c1.get_node_shared_ptr()->get_friendly_name());

            std::shared_ptr<ov::Node> matched_result;
            if (pattern_to_output.count(result1)) {
                matched_result = pattern_to_output.at(result1).get_node_shared_ptr();
            } else if (pattern_to_output.count(result2)) {
                matched_result = pattern_to_output.at(result2).get_node_shared_ptr();
                // TODO: need to check that upscale * downscale = 1
                // TODO: need to check input type is f8e5m2 or f8e4m3 if we use this version of concat
            }
            LOG_DEBUG(m.get_name() << ": matched_result=" << matched_result->get_friendly_name());

            matched_result->inputs()[0].replace_source_output(c1);

            return true;
        };

        register_matcher(std::make_shared<opp::Matcher>(result_or, "RedirectNewKvToOutputMatcher"), callback);
    }
};

namespace {

bool redirect_new_kv_to_output(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager("redirect_new_kv_to_output");
    manager.register_pass<RedirectNewKvToOutputMatcher>();
    bool result = manager.run_passes(model);
    model->validate_nodes_and_infer_types();

    return result;
}

}  // namespace

bool RedirectNewKvToOutput::run_on_model(const std::shared_ptr<ov::Model>& model) {
    return redirect_new_kv_to_output(model);
}
