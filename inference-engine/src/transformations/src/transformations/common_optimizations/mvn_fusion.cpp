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
#include <ngraph/pattern/op/or.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::MVNFusion, "MVNFusion", 0);

NGRAPH_RTTI_DEFINITION(ngraph::pass::MVNFusionOutsideSqrt, "MVNFusionOutsideSqrt", 0);

template <class T>
std::function<bool(Output<Node>)> value_is_equal_to(const std::vector<T>& ref_values)
{
    return [=](Output<Node> output) -> bool {
        auto node = output.get_node_shared_ptr();
        if (auto const_node = std::dynamic_pointer_cast<ngraph::op::Constant>(node)) {
            return const_node->template cast_vector<T>() == ref_values;
        }
        return false;
    };
}

ngraph::pass::MVNFusionOutsideSqrt::MVNFusionOutsideSqrt() {
    MATCHER_SCOPE(MVNFusionOutsideSqrt);
    // Detect MVN decomposition pattern:
    // (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps)
    auto x = pattern::any_input();

    // (x - ReduceMean(x, axes))
    //     `------mean1-------'
    auto mean1_axes = pattern::wrap_type<opset6::Constant>();
    auto mean1 = pattern::wrap_type<opset6::ReduceMean>({ x, mean1_axes });

    // (x - ReduceMean(x, axes))
    // `-sub1------------------'
    auto sub1 = pattern::wrap_type<opset6::Subtract>({ x, mean1 });

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                     `---mean2----------'
    auto mean2_axes = pattern::wrap_type<opset6::Constant>();
    auto mean2 = pattern::wrap_type<opset6::ReduceMean>({ x, mean2_axes });

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                 `-sub2------------------'
    auto sub2 = pattern::wrap_type<opset6::Subtract>({ x, mean2 });

    const auto reuseSub1OrNot = std::make_shared<pattern::op::Or>(OutputVector{ sub1, sub2 });

    auto cast = pattern::wrap_type<opset6::Convert>({ reuseSub1OrNot });
    const auto hasConvertOrNot = std::make_shared<pattern::op::Or>(OutputVector{ cast, reuseSub1OrNot });

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                 `---------------------power--'
    auto const_2 = pattern::wrap_type<opset6::Constant>(value_is_equal_to<float>({ 2.0 }));
    auto power = pattern::wrap_type<opset6::Power>({ hasConvertOrNot, const_2 });

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //     `---mean3--------------------------------'
    auto mean3_axes = pattern::wrap_type<opset6::Constant>();
    auto mean3 = pattern::wrap_type<opset6::ReduceMean>({ power, mean3_axes });

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    // `--Power--------------------------------------'
    auto const_0_5 = pattern::wrap_type<ngraph::opset6::Constant>(value_is_equal_to<float>({0.5}));
    auto power_sqrt = pattern::wrap_type<opset6::Power>({ mean3, const_0_5 });

    auto sqrt = pattern::wrap_type<opset6::Sqrt>({ mean3 });

    const auto powerOrSqrt = std::make_shared<pattern::op::Or>(OutputVector{ power_sqrt, sqrt });

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps
    // `-----------------------------------------------Add---'
    auto eps = pattern::wrap_type<opset6::Constant>();
    auto add_eps = pattern::wrap_type<opset6::Add>({ powerOrSqrt, eps });

    // Final Divide
    auto const_neg_1 = pattern::wrap_type<opset6::Constant>(value_is_equal_to<float>({ -1 }));
    auto power_div = pattern::wrap_type<opset6::Power>({ add_eps, const_neg_1 });
    auto div = pattern::wrap_type<opset6::Multiply>({ sub1, power_div });

    auto div_alt = pattern::wrap_type<opset6::Divide>({ sub1, add_eps });
    const auto powerMulOrDiv = std::make_shared<pattern::op::Or>(OutputVector{ div, div_alt });

    ngraph::matcher_pass_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto exp_input = pattern_to_output.at(x);

        auto const_eps_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(eps).get_node_shared_ptr());
        float eps_value;
        if (!op::util::get_single_value(const_eps_node, eps_value)) {
            return false;
        }

        auto axes_1_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(mean1_axes).get_node_shared_ptr());
        auto axes_3_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(mean3_axes).get_node_shared_ptr());

        if (!axes_1_node || !axes_3_node) {
            return false;
        }

        auto axes_1_value = axes_1_node->cast_vector<int64_t>();
        auto axes_3_value = axes_3_node->cast_vector<int64_t>();

        if (axes_1_value != axes_3_value) {
            return false;
        }
        if (pattern_to_output.count(mean2_axes)) {
            auto axes_2_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(mean2_axes).get_node_shared_ptr());
            if (!axes_2_node) {
                return false;
            }
            auto axes_2_value = axes_2_node->cast_vector<int64_t>();
            if (!(axes_1_value == axes_2_value)) {
                return false;
            }
        }

        auto mvn = std::make_shared<ngraph::opset6::MVN>(exp_input, axes_1_node, true, eps_value, op::MVNEpsMode::OUTSIDE_SQRT);

        ngraph::NodeVector nodes_to_copy_info({ pattern_to_output.at(mean1).get_node_shared_ptr(),
                                                pattern_to_output.at(sub1).get_node_shared_ptr(),
                                                pattern_to_output.at(power).get_node_shared_ptr(),
                                                pattern_to_output.at(mean3).get_node_shared_ptr(),
                                                pattern_to_output.at(add_eps).get_node_shared_ptr() });

        if (pattern_to_output.count(mean2) && pattern_to_output.count(sub2)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(mean2).get_node_shared_ptr());
            nodes_to_copy_info.push_back(pattern_to_output.at(sub2).get_node_shared_ptr());
        }

        if (pattern_to_output.count(cast)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(cast).get_node_shared_ptr());
        }

        if (pattern_to_output.count(power_sqrt)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(power_sqrt).get_node_shared_ptr());
        } else if (pattern_to_output.count(sqrt)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(sqrt).get_node_shared_ptr());
        }

        if (pattern_to_output.count(div_alt)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(div_alt).get_node_shared_ptr());
        } else if (pattern_to_output.count(power_div) != 0 && pattern_to_output.count(div) != 0) {
            nodes_to_copy_info.push_back(pattern_to_output.at(power_div).get_node_shared_ptr());
            nodes_to_copy_info.push_back(pattern_to_output.at(div).get_node_shared_ptr());
        }

        mvn->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(nodes_to_copy_info, mvn);
        ngraph::replace_node(m.get_match_root(), mvn);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(powerMulOrDiv, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::MVNFusionInsideSqrt, "MVNFusionInsideSqrt", 0);

ngraph::pass::MVNFusionInsideSqrt::MVNFusionInsideSqrt() {
    MATCHER_SCOPE(MVNFusionInsideSqrt);
    auto single_consumer = pattern::consumers_count(1);

    // Detect MVN decomposition pattern:
    // (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps))
    auto x = pattern::any_input();

    // (x - ReduceMean(x, axes))
    //     `------mean1-------'
    auto mean1_axes = pattern::wrap_type<opset6::Constant>();
    auto mean1 = pattern::wrap_type<opset6::ReduceMean>({ x, mean1_axes }, single_consumer);

    // (x - ReduceMean(x, axes))
    // `-sub1------------------'
    auto sub1 = pattern::wrap_type<opset6::Subtract>({ x, mean1 });

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                     `---mean2----------'
    auto mean2_axes = pattern::wrap_type<opset6::Constant>();
    auto mean2 = pattern::wrap_type<opset6::ReduceMean>({ x, mean2_axes }, single_consumer);

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                 `-sub2------------------'
    auto sub2 = pattern::wrap_type<opset6::Subtract>({ x, mean2 }, single_consumer);

    const auto reuseSub1OrNot = std::make_shared<pattern::op::Or>(OutputVector{ sub1, sub2 });

    auto cast = pattern::wrap_type<opset6::Convert>({ reuseSub1OrNot }, single_consumer);
    const auto hasConvertOrNot = std::make_shared<pattern::op::Or>(OutputVector{ cast, reuseSub1OrNot });

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                 `---------------------power--'
    auto const_2 = pattern::wrap_type<opset6::Constant>();
    auto power = pattern::wrap_type<opset6::Power>({ hasConvertOrNot, const_2 }, single_consumer);

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //     `---mean3--------------------------------'
    auto mean3_axes = pattern::wrap_type<opset6::Constant>();
    auto mean3 = pattern::wrap_type<opset6::ReduceMean>({ power, mean3_axes }, single_consumer);

    // (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps))
    // `-----------------------------------------------Add---'
    auto eps = pattern::wrap_type<opset6::Constant>();
    auto add_eps = pattern::wrap_type<opset6::Add>({ mean3, eps }, single_consumer);

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    // `--Power--------------------------------------'
    auto const_0_5 = pattern::wrap_type<ngraph::opset6::Constant>();
    auto power_sqrt = pattern::wrap_type<opset6::Power>({ add_eps, const_0_5 }, single_consumer);

    auto sqrt = pattern::wrap_type<opset6::Sqrt>({ add_eps }, single_consumer);

    const auto powerOrSqrt = std::make_shared<pattern::op::Or>(OutputVector{ power_sqrt, sqrt });

    // Final Divide
    auto const_neg_1 = pattern::wrap_type<opset6::Constant>();
    auto power_div = pattern::wrap_type<opset6::Power>({ powerOrSqrt, const_neg_1 }, single_consumer);
    auto div = pattern::wrap_type<opset6::Multiply>({ sub1, power_div });

    auto div_alt = pattern::wrap_type<opset6::Divide>({ sub1, powerOrSqrt });
    const auto powerMulOrDiv = std::make_shared<pattern::op::Or>(OutputVector{ div, div_alt });

    ngraph::matcher_pass_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto exp_input = pattern_to_output.at(x);
        auto const_2_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(const_2).get_node_shared_ptr());

        if (!const_2_node) {
            return false;
        }

        auto const_eps_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(eps).get_node_shared_ptr());
        float eps_value;

        if (!op::util::has_constant_value<float>(const_2_node, 2.0) || !op::util::get_single_value(const_eps_node, eps_value)) {
            return false;
        }

        if (pattern_to_output.count(const_0_5)) {
            auto const_0_5_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(const_0_5).get_node_shared_ptr());

            if (!const_0_5_node || !op::util::has_constant_value<float>(const_0_5_node, 0.5)) {
                return false;
            }
        }

        if (pattern_to_output.count(const_neg_1)) {
            auto const_neg_1_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(const_neg_1).get_node_shared_ptr());

            if (!const_neg_1_node || !op::util::has_constant_value<float>(const_neg_1_node, -1)) {
                return false;
            }
        }

        auto axes_1_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(mean1_axes).get_node_shared_ptr());
        auto axes_3_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(mean3_axes).get_node_shared_ptr());

        if (!axes_1_node || !axes_3_node) {
            return false;
        }

        auto axes_1_value = axes_1_node->cast_vector<int64_t>();
        auto axes_3_value = axes_3_node->cast_vector<int64_t>();

        if (axes_1_value != axes_3_value) {
            return false;
        }
        if (pattern_to_output.count(mean2_axes)) {
            auto axes_2_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(mean2_axes).get_node_shared_ptr());
            if (!axes_2_node) {
                return false;
            }
            auto axes_2_value = axes_2_node->cast_vector<int64_t>();
            if (!(axes_1_value == axes_2_value)) {
                return false;
            }
        }

        auto mvn = std::make_shared<ngraph::opset6::MVN>(exp_input, axes_1_node, true, eps_value, op::MVNEpsMode::INSIDE_SQRT);

        ngraph::NodeVector nodes_to_copy_info({ pattern_to_output.at(mean1).get_node_shared_ptr(),
                                                pattern_to_output.at(sub1).get_node_shared_ptr(),
                                                pattern_to_output.at(power).get_node_shared_ptr(),
                                                pattern_to_output.at(mean3).get_node_shared_ptr(),
                                                pattern_to_output.at(add_eps).get_node_shared_ptr() });

        if (pattern_to_output.count(mean2) && pattern_to_output.count(sub2)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(mean2).get_node_shared_ptr());
            nodes_to_copy_info.push_back(pattern_to_output.at(sub2).get_node_shared_ptr());
        }

        if (pattern_to_output.count(cast)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(cast).get_node_shared_ptr());
        }

        if (pattern_to_output.count(power_sqrt)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(power_sqrt).get_node_shared_ptr());
        } else if (pattern_to_output.count(sqrt)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(sqrt).get_node_shared_ptr());
        }

        if (pattern_to_output.count(div_alt)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(div_alt).get_node_shared_ptr());
        } else if (pattern_to_output.count(power_div) && pattern_to_output.count(div)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(power_div).get_node_shared_ptr());
            nodes_to_copy_info.push_back(pattern_to_output.at(div).get_node_shared_ptr());
        }

        mvn->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(nodes_to_copy_info, mvn);
        ngraph::replace_node(m.get_match_root(), mvn);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(powerMulOrDiv, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
