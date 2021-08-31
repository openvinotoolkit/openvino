// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/random_uniform_fusion.hpp"

#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::RandomUniformFusion, "RandomUniformFusion",
                       0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::RandomUniformMaxValFusion,
                       "RandomUniformMaxValFusion", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::RandomUniformMinValFusion,
                       "RandomUniformMinValFusion", 0);

ngraph::pass::RandomUniformMaxValFusion::RandomUniformMaxValFusion() {
  MATCHER_SCOPE(RandomUniformMaxValFusion);
  auto data_pattern = ngraph::pattern::any_input();
  auto ru_min_input_pattern = ngraph::pattern::any_input();
  auto ru_max_input_pattern = ngraph::pattern::any_input();
  auto random_uniform_pattern =
      ngraph::pattern::wrap_type<opset8::RandomUniform>(
          {data_pattern, ru_min_input_pattern, ru_max_input_pattern},
          pattern::consumers_count(1));
  auto mul_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
  auto mul_pattern = ngraph::pattern::wrap_type<opset8::Multiply>(
      {random_uniform_pattern, mul_const_pattern});

  auto conv_pattern =
      ngraph::pattern::wrap_type<opset8::Convert>({random_uniform_pattern});
  auto mul_with_convert_pattern = ngraph::pattern::wrap_type<opset8::Multiply>(
      {conv_pattern, mul_const_pattern});

  auto mul_or_mul_with_convert_pattern = std::make_shared<pattern::op::Or>(
      OutputVector{mul_pattern, mul_with_convert_pattern});

  ngraph::matcher_pass_callback callback = [=](pattern::Matcher &m) {
    auto pattern_map = m.get_pattern_value_map();
    auto data = pattern_map[data_pattern];
    auto random_uniform = pattern_map[random_uniform_pattern];
    auto mul_const = pattern_map[mul_const_pattern];
    auto mul_or_mul_with_convert = pattern_map[mul_or_mul_with_convert_pattern];
    auto ru = std::dynamic_pointer_cast<opset8::RandomUniform>(
        random_uniform.get_node_shared_ptr());
    if (!ru)
      return false;

    if (pattern_map.count(conv_pattern)) {
      auto mul = pattern_map[mul_with_convert_pattern];
      auto ml = std::dynamic_pointer_cast<opset8::Multiply>(
          mul.get_node_shared_ptr());
      if (!ml)
        return false;

      auto conv = pattern_map[conv_pattern];
      auto cv = std::dynamic_pointer_cast<opset8::Convert>(
          conv.get_node_shared_ptr());
      if (!cv)
        return false;
      auto new_conv = register_new_node<ngraph::opset8::Convert>(
          ml->input_value(1), ru->get_out_type());
      auto new_mul = ml->clone_with_new_inputs({ru->input_value(2), new_conv});
      auto new_ru =
          ru->clone_with_new_inputs({data, ru->input_value(1), new_mul});
      new_ru->set_friendly_name(m.get_match_root()->get_friendly_name());
      auto new_ru_conv = cv->clone_with_new_inputs({new_ru});
      copy_runtime_info({ru, cv}, {new_ru, new_ru_conv});
      ngraph::replace_node(m.get_match_root(), new_ru_conv);

    } else {
      auto mul = pattern_map[mul_pattern];
      auto ml = std::dynamic_pointer_cast<opset8::Multiply>(
          mul.get_node_shared_ptr());
      if (!ml)
        return false;
      auto new_mul =
          ml->clone_with_new_inputs({ru->input_value(2), ml->input_value(1)});
      auto new_ru =
          ru->clone_with_new_inputs({data, ru->input_value(1), new_mul});
      new_ru->set_friendly_name(m.get_match_root()->get_friendly_name());
      copy_runtime_info({mul.get_node_shared_ptr(), ru}, {new_mul, new_ru});
      ngraph::replace_node(m.get_match_root(), new_ru);
    }

    return true;
  };

  auto m = std::make_shared<ngraph::pattern::Matcher>(
      mul_or_mul_with_convert_pattern, matcher_name);
  this->register_matcher(m, callback);
}

ngraph::pass::RandomUniformMinValFusion::RandomUniformMinValFusion() {
  MATCHER_SCOPE(RandomUniformMinValFusion);
  auto data_pattern = ngraph::pattern::any_input();
  auto ru_min_input_pattern = ngraph::pattern::any_input();
  auto ru_max_input_pattern = ngraph::pattern::any_input();
  auto random_uniform_pattern =
      ngraph::pattern::wrap_type<opset8::RandomUniform>(
          {data_pattern, ru_min_input_pattern, ru_max_input_pattern},
          pattern::consumers_count(1));
  auto add_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
  auto add_pattern = ngraph::pattern::wrap_type<opset8::Add>(
      {random_uniform_pattern, add_const_pattern});

  auto conv_pattern =
      ngraph::pattern::wrap_type<opset8::Convert>({random_uniform_pattern});
  auto add_with_convert_pattern = ngraph::pattern::wrap_type<opset8::Add>(
      {conv_pattern, add_const_pattern});

  auto add_or_add_with_convert_pattern = std::make_shared<pattern::op::Or>(
      OutputVector{add_pattern, add_with_convert_pattern});

  ngraph::matcher_pass_callback callback = [=](pattern::Matcher &m) {
    auto pattern_map = m.get_pattern_value_map();
    auto data = pattern_map[data_pattern];
    auto random_uniform = pattern_map[random_uniform_pattern];
    auto add_const = pattern_map[add_const_pattern];
    auto ru_max_input = pattern_map[ru_max_input_pattern];
    auto ru_min_input = pattern_map[ru_min_input_pattern];
    auto ru = std::dynamic_pointer_cast<opset8::RandomUniform>(
        random_uniform.get_node_shared_ptr());
    if (!ru)
      return false;

    if (pattern_map.count(conv_pattern)) {
      auto add = pattern_map[add_with_convert_pattern];
      auto conv = pattern_map[conv_pattern];
      auto cv = std::dynamic_pointer_cast<opset8::Convert>(
          conv.get_node_shared_ptr());
      if (!cv)
        return false;
      auto add_conv = register_new_node<ngraph::opset8::Convert>(
          add_const, ru->get_out_type());
      auto add2 =
          register_new_node<ngraph::opset8::Add>(add_conv, ru_max_input);
      auto add1 =
          register_new_node<ngraph::opset8::Add>(add_conv, ru_min_input);
      auto new_ru = ru->clone_with_new_inputs({data, add1, add2});
      new_ru->set_friendly_name(m.get_match_root()->get_friendly_name());
      auto new_ru_conv = cv->clone_with_new_inputs({new_ru});
      copy_runtime_info({add.get_node_shared_ptr(), ru, cv},
                        {new_ru, new_ru_conv});
      ngraph::replace_node(m.get_match_root(), new_ru_conv);
    } else {
      auto add = pattern_map[add_pattern];
      auto add2 =
          register_new_node<ngraph::opset8::Add>(add_const, ru_max_input);
      auto add1 =
          register_new_node<ngraph::opset8::Add>(add_const, ru_min_input);
      auto new_ru = ru->clone_with_new_inputs({data, add1, add2});
      new_ru->set_friendly_name(m.get_match_root()->get_friendly_name());
      copy_runtime_info({add.get_node_shared_ptr(), ru}, new_ru);
      ngraph::replace_node(m.get_match_root(), new_ru);
    }
    return true;
  };

  auto m = std::make_shared<ngraph::pattern::Matcher>(
      add_or_add_with_convert_pattern, matcher_name);
  this->register_matcher(m, callback);
}
