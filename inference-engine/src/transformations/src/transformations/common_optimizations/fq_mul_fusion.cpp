// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fq_mul_fusion.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

namespace {
  std::pair<ngraph::Output<ngraph::Node>, ngraph::Output<ngraph::Node>>
  get_adjusted_output_range(ngraph::Output<ngraph::Node> out_low,
                            ngraph::Output<ngraph::Node> out_high,
                            ngraph::Output<ngraph::Node> multiplier) {
    const auto mul_out_low = std::make_shared<ngraph::opset4::Multiply>(out_low, multiplier);
    const auto mul_out_high = std::make_shared<ngraph::opset4::Multiply>(out_high, multiplier);

    ngraph::OutputVector folded_out_low(1), folded_out_high(1);

    if (!mul_out_low->constant_fold(folded_out_low, {out_low, multiplier})) {
      throw ngraph::ngraph_error("Could not constant fold the output_low Multiply operation");
    }

    if (!mul_out_high->constant_fold(folded_out_high, {out_high, multiplier})) {
      throw ngraph::ngraph_error("Could not constant fold the output_high Multiply operation");
    }

    return {folded_out_low[0], folded_out_high[0]};
  }
} // namespace

// This transformation multiplies the "output_low" and "output_high" inputs of the FQ operation
// by the constant value that before transormation is used to multiply the output of FQ.
// The Multiply node is removed along with the constant (C6).
//
//            C1 C2 C3 C4 C5
//            |  |  |  |  |
//            |  |  |  |  |                     C1 C2 C3   C4*C6    C5*C6
//            v  v  v  v  v                     |  |  |      |        |
//          +--------------+                    |  |  |      |        |
//          | FakeQuantize |                    v  v  v      v        v
//          +--------------+                  +--------------------------+
//                 |                =====>    |       FakeQuantize       |
//                 v                          +--------------------------+
//            +----------+                                  |
//            | Multiply | <--- C6                          v
//            +----+-----+
//                 |
//                 v
//

ngraph::pass::FakeQuantizeMulFusion::FakeQuantizeMulFusion() {
  const auto fq_output_low_p = ngraph::pattern::wrap_type<opset4::Constant>();
  const auto fq_output_high_p = ngraph::pattern::wrap_type<opset4::Constant>();

  const auto fq_node_p = ngraph::pattern::wrap_type<opset4::FakeQuantize>(
      {ngraph::pattern::any_input(),
       ngraph::pattern::wrap_type<opset4::Constant>(),
       ngraph::pattern::wrap_type<opset4::Constant>(),
       fq_output_low_p,
       fq_output_high_p},
      pattern::consumers_count(1));

  const auto mul_constant_p = ngraph::pattern::wrap_type<opset4::Constant>();
  const auto mul_node_p = ngraph::pattern::wrap_type<opset4::Multiply>(
      {fq_node_p, mul_constant_p}, pattern::consumers_count(1));

  ngraph::matcher_pass_callback callback = [=](pattern::Matcher &m) {
    const auto& pattern_map = m.get_pattern_value_map();

    const auto fq_node = std::dynamic_pointer_cast<ngraph::opset4::FakeQuantize>(
        pattern_map.at(fq_node_p).get_node_shared_ptr());
    if (!fq_node) {
      return false;
    }

    const auto output_low_const = pattern_map.at(fq_output_low_p);
    const auto output_high_const = pattern_map.at(fq_output_high_p);
    const auto mul_constant = pattern_map.at(mul_constant_p);

    const auto new_output_limits = get_adjusted_output_range(
      output_low_const, output_high_const, mul_constant);

    const auto new_fq_node = fq_node->clone_with_new_inputs({fq_node->input_value(0),
                                                            fq_node->input_value(1),
                                                            fq_node->input_value(2),
                                                            new_output_limits.first,
                                                            new_output_limits.second});

    const auto mul_node = pattern_map.at(mul_node_p).get_node_shared_ptr();
    replace_node(mul_node, new_fq_node);

    new_fq_node->set_friendly_name(fq_node->get_friendly_name());
    copy_runtime_info({fq_node, mul_node}, new_fq_node);

    return true;
  };

  auto m = std::make_shared<ngraph::pattern::Matcher>(mul_node_p,
                                                      "FakeQuantizeMulFusion");
  this->register_matcher(m, callback);
}
