// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/random_uniform_fusion.hpp"

#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::RandomUniformFusion, "RandomUniformFusion", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::RandomUniformMaxValFusion, "RandomUniformMaxValFusion", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::RandomUniformMinValFusion, "RandomUniformMinValFusion", 0);

ngraph::pass::RandomUniformMaxValFusion::RandomUniformMaxValFusion() {
    MATCHER_SCOPE(RandomUniformMaxValFusion);
    auto data_pattern = ngraph::pattern::any_input();
    auto ru_min_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
    auto ru_max_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
    auto random_uniform_pattern =
        ngraph::pattern::wrap_type<opset8::RandomUniform>({data_pattern, ru_min_const_pattern, ru_max_const_pattern},
                                                          pattern::consumers_count(1));
    auto mul_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
    auto mul_pattern = ngraph::pattern::wrap_type<opset8::Multiply>({random_uniform_pattern, mul_const_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto data = pattern_map[data_pattern];
        auto random_uniform = pattern_map[random_uniform_pattern];
        auto ru_max_val_const = pattern_map[ru_max_const_pattern];
        auto mul = pattern_map[mul_pattern];
        auto mul_const = pattern_map[mul_const_pattern];
        auto ru = std::dynamic_pointer_cast<opset8::RandomUniform>(random_uniform.get_node_shared_ptr());
        if (!ru)
            return false;
        if (!ru->get_out_type().is_real())
            return false;

        auto max_val_const = std::dynamic_pointer_cast<opset8::Constant>(ru_max_val_const.get_node_shared_ptr());
        if (!max_val_const)
            return false;
        auto max_value = max_val_const->cast_vector<float>()[0];
        if (max_value != 1.0f)
            return false;

        auto new_ru = ru->clone_with_new_inputs({data, ru->input_value(1), mul_const});
        new_ru->set_friendly_name(m.get_match_root()->get_friendly_name());
        copy_runtime_info({mul.get_node_shared_ptr(), ru}, new_ru);
        ngraph::replace_node(m.get_match_root(), new_ru);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul_pattern, matcher_name);
    this->register_matcher(m, callback);
}

ngraph::pass::RandomUniformMinValFusion::RandomUniformMinValFusion() {
    MATCHER_SCOPE(RandomUniformMinValFusion);
    auto data_pattern = ngraph::pattern::any_input();
    auto ru_min_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
    auto ru_max_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
    auto random_uniform_pattern =
        ngraph::pattern::wrap_type<opset8::RandomUniform>({data_pattern, ru_min_const_pattern, ru_max_const_pattern},
                                                          pattern::consumers_count(1));
    auto add_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
    auto add_pattern = ngraph::pattern::wrap_type<opset8::Add>({random_uniform_pattern, add_const_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto data = pattern_map[data_pattern];
        auto random_uniform = pattern_map[random_uniform_pattern];
        auto add = pattern_map[add_pattern];
        auto add_const = pattern_map[add_const_pattern];
        auto ru_max_const = pattern_map[ru_max_const_pattern];
        auto ru_min_val_const = pattern_map[ru_min_const_pattern];
        auto ru = std::dynamic_pointer_cast<opset8::RandomUniform>(random_uniform.get_node_shared_ptr());
        if (!ru)
            return false;
        if (!ru->get_out_type().is_real())
            return false;

        auto min_val_const = std::dynamic_pointer_cast<opset8::Constant>(ru_min_val_const.get_node_shared_ptr());
        if (!min_val_const)
            return false;
        auto max_value = min_val_const->cast_vector<float>()[0];
        if (max_value != 0.0f)
            return false;

        auto new_add = register_new_node<ngraph::opset8::Add>(add_const, ru_max_const);
        auto new_ru = ru->clone_with_new_inputs({data, add_const, new_add});
        new_ru->set_friendly_name(m.get_match_root()->get_friendly_name());
        copy_runtime_info({add.get_node_shared_ptr(), ru}, new_ru);
        ngraph::replace_node(m.get_match_root(), new_ru);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add_pattern, matcher_name);
    this->register_matcher(m, callback);
}
