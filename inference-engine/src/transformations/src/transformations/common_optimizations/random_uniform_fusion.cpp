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
  const auto data_pattern = ngraph::pattern::any_input();
  const auto ru_min_input_pattern = ngraph::pattern::any_input();
  const auto ru_max_input_pattern = ngraph::pattern::any_input();
  const auto random_uniform_pattern =
      ngraph::pattern::wrap_type<opset8::RandomUniform>(
          {data_pattern, ru_min_input_pattern, ru_max_input_pattern},
          pattern::consumers_count(1));
  const auto mul_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
  const auto mul_pattern = ngraph::pattern::wrap_type<opset8::Multiply>(
      {random_uniform_pattern, mul_const_pattern});

  const auto conv_pattern =
      ngraph::pattern::wrap_type<opset8::Convert>({random_uniform_pattern});
  const auto mul_with_convert_pattern =
      ngraph::pattern::wrap_type<opset8::Multiply>(
          {conv_pattern, mul_const_pattern});

  const auto mul_or_mul_with_convert_pattern =
      std::make_shared<pattern::op::Or>(
          OutputVector{mul_pattern, mul_with_convert_pattern});

  ngraph::matcher_pass_callback callback = [=](pattern::Matcher &m) {
    auto pattern_map = m.get_pattern_value_map();
    const auto data = pattern_map[data_pattern];
    const auto random_uniform = pattern_map[random_uniform_pattern];
    const auto mul_const = pattern_map[mul_const_pattern];
    const auto mul_or_mul_with_convert =
        pattern_map[mul_or_mul_with_convert_pattern];
    const auto ru = std::dynamic_pointer_cast<opset8::RandomUniform>(
        random_uniform.get_node_shared_ptr());
    if (!ru)
      return false;

    if (pattern_map.count(conv_pattern)) {
      const auto mul = pattern_map[mul_with_convert_pattern];
      const auto ml = std::dynamic_pointer_cast<opset8::Multiply>(
          mul.get_node_shared_ptr());
      if (!ml)
        return false;

      const auto conv = pattern_map[conv_pattern];
      const auto cv = std::dynamic_pointer_cast<opset8::Convert>(
          conv.get_node_shared_ptr());
      if (!cv)
        return false;
      const auto new_conv = register_new_node<ngraph::opset8::Convert>(
          ml->input_value(1), ru->get_out_type());
      const auto new_mul =
          ml->clone_with_new_inputs({ru->input_value(2), new_conv});
      const auto new_ru =
          ru->clone_with_new_inputs({data, ru->input_value(1), new_mul});
      new_ru->set_friendly_name(m.get_match_root()->get_friendly_name());
      const auto new_ru_conv = cv->clone_with_new_inputs({new_ru});
      copy_runtime_info({ru, cv}, {new_ru, new_ru_conv});
      ngraph::replace_node(m.get_match_root(), new_ru_conv);

    } else {
      const auto mul = pattern_map[mul_pattern];
      const auto ml = std::dynamic_pointer_cast<opset8::Multiply>(
          mul.get_node_shared_ptr());
      if (!ml)
        return false;
      const auto new_mul =
          ml->clone_with_new_inputs({ru->input_value(2), ml->input_value(1)});
      const auto new_ru =
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
  const auto data_pattern = ngraph::pattern::any_input();
  const auto ru_min_input_pattern = ngraph::pattern::any_input();
  const auto ru_max_input_pattern = ngraph::pattern::any_input();
  const auto random_uniform_pattern =
      ngraph::pattern::wrap_type<opset8::RandomUniform>(
          {data_pattern, ru_min_input_pattern, ru_max_input_pattern},
          pattern::consumers_count(1));
  const auto add_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
  const auto add_pattern = ngraph::pattern::wrap_type<opset8::Add>(
      {random_uniform_pattern, add_const_pattern});

  const auto conv_pattern =
      ngraph::pattern::wrap_type<opset8::Convert>({random_uniform_pattern});
  const auto add_with_convert_pattern = ngraph::pattern::wrap_type<opset8::Add>(
      {conv_pattern, add_const_pattern});

  const auto add_or_add_with_convert_pattern =
      std::make_shared<pattern::op::Or>(
          OutputVector{add_pattern, add_with_convert_pattern});

  ngraph::matcher_pass_callback callback = [=](pattern::Matcher &m) {
    auto pattern_map = m.get_pattern_value_map();
    const auto data = pattern_map[data_pattern];
    const auto random_uniform = pattern_map[random_uniform_pattern];
    const auto add_const = pattern_map[add_const_pattern];
    const auto ru_max_input = pattern_map[ru_max_input_pattern];
    const auto ru_min_input = pattern_map[ru_min_input_pattern];
    const auto ru = std::dynamic_pointer_cast<opset8::RandomUniform>(
        random_uniform.get_node_shared_ptr());
    if (!ru)
      return false;

    if (pattern_map.count(conv_pattern)) {
      const auto add = pattern_map[add_with_convert_pattern];
      const auto conv = pattern_map[conv_pattern];
      const auto cv = std::dynamic_pointer_cast<opset8::Convert>(
          conv.get_node_shared_ptr());
      if (!cv)
        return false;
      const auto add_conv = register_new_node<ngraph::opset8::Convert>(
          add_const, ru->get_out_type());
      const auto add2 =
          register_new_node<ngraph::opset8::Add>(add_conv, ru_max_input);
      const auto add1 =
          register_new_node<ngraph::opset8::Add>(add_conv, ru_min_input);
      const auto new_ru = ru->clone_with_new_inputs({data, add1, add2});
      new_ru->set_friendly_name(m.get_match_root()->get_friendly_name());
      const auto new_ru_conv = cv->clone_with_new_inputs({new_ru});
      copy_runtime_info({add.get_node_shared_ptr(), ru, cv},
                        {new_ru, new_ru_conv});
      ngraph::replace_node(m.get_match_root(), new_ru_conv);
    } else {
      const auto add = pattern_map[add_pattern];
      const auto add2 =
          register_new_node<ngraph::opset8::Add>(add_const, ru_max_input);
      const auto add1 =
          register_new_node<ngraph::opset8::Add>(add_const, ru_min_input);
      const auto new_ru = ru->clone_with_new_inputs({data, add1, add2});
      new_ru->set_friendly_name(m.get_match_root()->get_friendly_name());
      copy_runtime_info({add.get_node_shared_ptr(), ru}, new_ru);
      ngraph::replace_node(m.get_match_root(), new_ru);
    }
    return true;
  };

  const auto m = std::make_shared<ngraph::pattern::Matcher>(
      add_or_add_with_convert_pattern, matcher_name);
  this->register_matcher(m, callback);
}
