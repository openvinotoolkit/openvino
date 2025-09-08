// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "template_pattern_transformation.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "template_model_transformation.hpp"

// ! [graph_rewrite:template_transformation_cpp]
// template_pattern_transformation.cpp
ov::pass::DecomposeDivideMatcher::DecomposeDivideMatcher() {
    MATCHER_SCOPE(DecomposeDivideMatcher);
    // Pattern example
    auto input0 = pattern::any_input();
    auto input1 = pattern::any_input();
    auto div = std::make_shared<ov::opset3::Divide>(input0, input1);

    ov::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto div = ov::as_type_ptr<ov::opset3::Divide>(m.get_match_root());
        // We can not apply this transformation in case with integer input data type
        if (!div || div->input(0).get_element_type().is_integral()) {
            return false;
        }

        // Decompose Divide into Multiply with Power operations
        auto pow = std::make_shared<ov::opset3::Power>(
            div->input_value(1),
            opset3::Constant::create(div->get_input_element_type(1), Shape{1}, {-1}));

        auto mul = std::make_shared<ov::opset3::Multiply>(div->input_value(0), pow);

        // Save original name to last operation in replacement sub-graph
        mul->set_friendly_name(div->get_friendly_name());

        // Copy runtime info attributes to newly created operation
        ov::copy_runtime_info(div, {pow, mul});

        // Replace Divide operation with Multiply
        ov::replace_node(div, mul);

        // Return true as the root node was changed
        return true;
    };

    // Register pattern with Divide operation as a pattern root node
    auto m = std::make_shared<ov::pass::pattern::Matcher>(div, "ConvertDivide");
    // Register Matcher
    register_matcher(m, callback);
}
// ! [graph_rewrite:template_transformation_cpp]

// ! [matcher_pass:relu_fusion]
ov::pass::ReluReluFusionMatcher::ReluReluFusionMatcher() {
    MATCHER_SCOPE(ReluReluFusionMatcher);
    auto m_relu1 = ov::pass::pattern::wrap_type<ov::opset3::Relu>(pattern::consumers_count(1));
    auto m_relu2 = ov::pass::pattern::wrap_type<ov::opset3::Relu>({m_relu1});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        // Map that helps to connect labels with matched outputs
        auto& node_to_output = m.get_pattern_value_map();

        // Create new Relu operation and add register it for additional execution
        auto new_relu =
            register_new_node<ov::opset3::Relu>(node_to_output.at(m_relu1).get_node_shared_ptr()->input_value(0));

        // Copy runtime info attributes to newly created operation
        ov::copy_runtime_info(m.get_matched_nodes(), new_relu);

        // Save last Relu name to new Relu operation
        new_relu->set_friendly_name(m.get_match_root()->get_friendly_name());

        // Replace Relu->Relu with Relu
        ov::replace_node(m.get_match_root(), new_relu);

        // Return true as the root node was changed
        return true;
    };

    // Register pattern with Relu operation as a pattern root node
    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_relu2, "ReluReluFusion");
    // Register Matcher
    register_matcher(m, callback);
}
// ! [matcher_pass:relu_fusion]

void run_matcher_on_node(std::shared_ptr<ov::Node> node) {
    // ! [matcher_pass:run_on_node]
    if (ov::pass::DecomposeDivideMatcher().apply(node)) {
        // successful execution (root node was replaced)
    }
    // ! [matcher_pass:run_on_node]
}

void run_matcher_with_manager(std::shared_ptr<ov::Model> f) {
    // ! [matcher_pass:manager]
    // Two matchers will run independently (two independent graph traversals)
    // pass::Manager automatically creates GraphRewrite container for each MatcherPass
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::DecomposeDivideMatcher>();
    manager.register_pass<ov::pass::ReluReluFusionMatcher>();
    manager.run_passes(f);
    // ! [matcher_pass:manager]
}

void run_matcher_with_manager2(std::shared_ptr<ov::Model> f) {
    // ! [matcher_pass:manager2]
    // Register anchor GraphRewrite pass inside manager that will execute two matchers simultaneously
    ov::pass::Manager manager;
    auto anchor = manager.register_pass<ov::pass::GraphRewrite>();
    using namespace ov::pass;
    ADD_MATCHER(anchor, DecomposeDivideMatcher)
    ADD_MATCHER(anchor, ReluReluFusionMatcher)
    manager.run_passes(f);
    // ! [matcher_pass:manager2]
}

void run_matcher_with_manager3(std::shared_ptr<ov::Model> f) {
    // ! [matcher_pass:manager3]
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::MyModelTransformation>();
    // Two matchers will run independently (two independent graph traversals)
    // pass::Manager automatically creates GraphRewrite container for each MatcherPass
    manager.register_pass<ov::pass::DecomposeDivideMatcher>();
    manager.register_pass<ov::pass::ReluReluFusionMatcher>();
    manager.run_passes(f);
    // ! [matcher_pass:manager3]
}

void run_matcher_with_gr(std::shared_ptr<ov::Model> f) {
    // ! [matcher_pass:graph_rewrite]
    // Two matcher passes will run simultaneously in a single graph traversal
    ov::pass::GraphRewrite pass;
    pass.add_matcher<ov::pass::DecomposeDivideMatcher>();
    pass.add_matcher<ov::pass::ReluReluFusionMatcher>();
    pass.run_on_model(f);
    // ! [matcher_pass:graph_rewrite]
}

// ! [manual_constant_folding]
template <class T>
ov::Output<ov::Node> eltwise_fold(const ov::Output<ov::Node>& input0, const ov::Output<ov::Node>& input1) {
    auto eltwise = std::make_shared<T>(input0, input1);
    ov::OutputVector output(eltwise->get_output_size());
    // If constant folding wasn't successful return eltwise output
    if (!eltwise->constant_fold(output, {input0, input1})) {
        return eltwise->output(0);
    }
    return output[0];
}
// ! [manual_constant_folding]
