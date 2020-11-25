// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/normalize_l2_fusion.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::NormalizeL2Fusion, "NormalizeL2Fusion", 0);

NGRAPH_RTTI_DEFINITION(ngraph::pass::NormalizeL2FusionWithMax, "NormalizeL2FusionWithMax", 0);

ngraph::pass::NormalizeL2FusionWithMax::NormalizeL2FusionWithMax() {
    auto input = ngraph::pattern::any_input();

    auto exp = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto pow = std::make_shared<ngraph::opset4::Power>(input, exp);
    auto axes = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto reduce_sum = std::make_shared<ngraph::opset4::ReduceSum>(pow, axes);
    auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(reduce_sum);
    auto eps_const = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto sqrt_max_eps = std::make_shared<ngraph::opset4::Maximum>(sqrt, eps_const);
    auto divide = std::make_shared<ngraph::opset4::Divide>(input, sqrt_max_eps);

    ngraph::graph_rewrite_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();

        const auto data_input = pattern_to_output.at(input);
        const auto exp_input = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(exp).get_node_shared_ptr());
        const auto axes_input = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(axes).get_node_shared_ptr());
        const auto eps_attr = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(eps_const).get_node_shared_ptr());

        if (!exp_input || !axes_input || !eps_attr) {
            return false;
        }

        const bool is_square_pow = op::util::has_constant_value<float>(exp_input, 2.0f);
        if (!is_square_pow) {
            return false;
        }
        if (shape_size(eps_attr->get_shape()) > 1) {
            return false;
        }
        const auto eps_attr_value = eps_attr->cast_vector<float>()[0];

        auto normalize_l2 = std::make_shared<ngraph::opset4::NormalizeL2>(data_input, axes_input, eps_attr_value, op::EpsMode::MAX);

        normalize_l2->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(pow).get_node_shared_ptr(),
                                   pattern_to_output.at(reduce_sum).get_node_shared_ptr(),
                                   pattern_to_output.at(sqrt).get_node_shared_ptr(),
                                   pattern_to_output.at(sqrt_max_eps).get_node_shared_ptr(),
                                   pattern_to_output.at(divide).get_node_shared_ptr()
                                   },
                                   normalize_l2);
        ngraph::replace_node(m.get_match_root(), normalize_l2);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(divide, "NormalizeL2FusionWithMax");
    register_matcher(m, matcher_pass_callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::NormalizeL2FusionWithAdd, "NormalizeL2FusionWithAdd", 0);

ngraph::pass::NormalizeL2FusionWithAdd::NormalizeL2FusionWithAdd() {
    auto input = ngraph::pattern::any_input();

    auto exp = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto pow = std::make_shared<ngraph::opset4::Power>(input, exp);
    auto axes = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto reduce_sum = std::make_shared<ngraph::opset4::ReduceSum>(pow, axes);
    auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(reduce_sum);
    auto eps_const = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto sqrt_add_eps = std::make_shared<ngraph::opset4::Add>(sqrt, eps_const);
    auto divide = std::make_shared<ngraph::opset4::Divide>(input, sqrt_add_eps);

    ngraph::graph_rewrite_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();

        const auto data_input = pattern_to_output.at(input);
        const auto exp_input = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(exp).get_node_shared_ptr());
        const auto axes_input = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(axes).get_node_shared_ptr());
        const auto eps_attr = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(eps_const).get_node_shared_ptr());

        if (!exp_input || !axes_input || !eps_attr) {
            return false;
        }

        const bool is_square_pow = shape_size(exp_input->get_shape()) <= 1 && exp_input->cast_vector<int64_t>()[0] == 2;
        if (!is_square_pow) {
            return false;
        }
        if (shape_size(eps_attr->get_shape()) > 1) {
            return false;
        }
        const auto eps_attr_value = op::util::has_constant_value<float>(exp_input, 2.0f);

        auto normalize_l2 = std::make_shared<ngraph::opset4::NormalizeL2>(data_input, axes_input, eps_attr_value, op::EpsMode::ADD);

        normalize_l2->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(pow).get_node_shared_ptr(),
                                   pattern_to_output.at(reduce_sum).get_node_shared_ptr(),
                                   pattern_to_output.at(sqrt).get_node_shared_ptr(),
                                   pattern_to_output.at(sqrt_add_eps).get_node_shared_ptr(),
                                   pattern_to_output.at(divide).get_node_shared_ptr()
                                   },
                                   normalize_l2);
        ngraph::replace_node(m.get_match_root(), normalize_l2);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(divide, "NormalizeL2FusionWithMax");
    register_matcher(m, matcher_pass_callback);
}
