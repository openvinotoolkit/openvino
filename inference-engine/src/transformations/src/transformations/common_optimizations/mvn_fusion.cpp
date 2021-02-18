// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/common_optimizations/mvn_fusion.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset6.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::MVNFusion, "MVNFusion", 0);

ngraph::pass::MVNFusion::MVNFusion() {
    MATCHER_SCOPE(MVNFusion);
    auto single_consumer = pattern::consumers_count(1);

    // Detect MVN decomposition pattern:
    // (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps)
    auto x = pattern::any_input();

    // (x - ReduceMean(x, axes))
    //     `------mean1-------'
    auto mean1_axes = pattern::wrap_type<opset6::Constant>();
    auto mean1 = pattern::wrap_type<opset6::ReduceMean>({ x, mean1_axes }, single_consumer);

    // (x - ReduceMean(x, axes))
    // `-sub1------------------'
    auto sub1 = pattern::wrap_type<opset6::Subtract>({ x, mean1 }, single_consumer);

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                     `---mean2----------'
    auto mean2_axes = pattern::wrap_type<opset6::Constant>();
    auto mean2 = pattern::wrap_type<opset6::ReduceMean>({ x, mean2_axes }, single_consumer);

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                 `-sub2------------------'
    auto sub2 = pattern::wrap_type<opset6::Subtract>({ x, mean2 }, single_consumer);

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                 `---------------------power--'
    auto const_2 = pattern::wrap_type<opset6::Constant>();
    auto power = pattern::wrap_type<opset6::Power>({ sub2, const_2 }, single_consumer);

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //     `---mean3--------------------------------'
    auto mean3_axes = pattern::wrap_type<opset6::Constant>();
    auto mean3 = pattern::wrap_type<opset6::ReduceMean>({ power, mean3_axes }, single_consumer);

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    // `--Power--------------------------------------'
    auto const_0_5 = pattern::wrap_type<ngraph::opset6::Constant>();
    auto power_sqrt = pattern::wrap_type<opset6::Power>({ mean3, const_0_5 }, single_consumer);
    // TODO: use Or to accept opset6::Sqrt operation also.

    // (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps)
    // `-----------------------------------------------Add---'
    auto eps = pattern::wrap_type<opset6::Constant>();
    auto add_eps = pattern::wrap_type<opset6::Add>({ power_sqrt, eps }, single_consumer);

    // Final Divide
    auto const_neg_1 = pattern::wrap_type<opset6::Constant>();
    auto power_div = pattern::wrap_type<opset6::Power>({ add_eps, const_neg_1 }, single_consumer);
    auto div = pattern::wrap_type<opset6::Multiply>({ sub1, power_div });

    // TODO: use Or to accept opset6::Divide operation. Also as root operation has multiple types
    // we must handle it in GraphRewrite engine to perform efficient matching.
    ngraph::matcher_pass_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto exp_input = pattern_to_output.at(x);

        auto const_2_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(const_2).get_node_shared_ptr());
        auto const_0_5_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(const_0_5).get_node_shared_ptr());
        auto const_neg_1_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(const_neg_1).get_node_shared_ptr());

        if (!const_2_node || !const_0_5_node || !const_neg_1_node) {
            return false;
        }

        auto const_eps_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(eps).get_node_shared_ptr());
        float eps_value;

        bool valid_constant_values = op::util::has_constant_value<float>(const_2_node, 2.0)
            && op::util::has_constant_value<float>(const_0_5_node, 0.5)
            && op::util::has_constant_value<float>(const_neg_1_node, -1.0)
            && op::util::get_single_value(const_eps_node, eps_value);

        if (!valid_constant_values) {
            return false;
        }

        auto axes_1_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(mean1_axes).get_node_shared_ptr());
        auto axes_2_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(mean2_axes).get_node_shared_ptr());
        auto axes_3_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(mean3_axes).get_node_shared_ptr());

        if (!axes_1_node || !axes_2_node || !axes_3_node) {
            return false;
        }

        auto axes_1_value = axes_1_node->cast_vector<int64_t>();
        auto axes_2_value = axes_2_node->cast_vector<int64_t>();
        auto axes_3_value = axes_3_node->cast_vector<int64_t>();

        if (!(axes_1_value == axes_2_value && axes_2_value == axes_3_value)) {
            return false;
        }

        auto mvn = std::make_shared<ngraph::opset6::MVN>(exp_input, axes_1_node, true, eps_value, op::MVNEpsMode::OUTSIDE_SQRT);

        mvn->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({ pattern_to_output.at(mean1).get_node_shared_ptr(),
                                   pattern_to_output.at(sub1).get_node_shared_ptr(),
                                   pattern_to_output.at(mean2).get_node_shared_ptr(),
                                   pattern_to_output.at(sub2).get_node_shared_ptr(),
                                   pattern_to_output.at(power).get_node_shared_ptr(),
                                   pattern_to_output.at(mean3).get_node_shared_ptr(),
                                   pattern_to_output.at(power_sqrt).get_node_shared_ptr(),
                                   pattern_to_output.at(add_eps).get_node_shared_ptr(),
                                   pattern_to_output.at(power_div).get_node_shared_ptr(),
                                   pattern_to_output.at(div).get_node_shared_ptr() }, mvn);
        ngraph::replace_node(m.get_match_root(), mvn);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(div, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
