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

// This transformation multiplies the "output_low" and "output_high" inputs of the FQ operation
// by the constant value that before transormation is used to multiply the output of FQ.
// Two new Multiply nodes share the original constant C6.
//
//             C1 C2 C3 C4 C5                   C1 C2 C3      C4                        C5
//             |  |  |  |  |                    |  |  |       |                         |
//             |  |  |  |  |                    |  |  |       v                         v
//             v  v  v  v  v                    |  |  |  +----------+              +----------+
//           +--------------+                   |  |  |  | Multiply | <--- C6 ---> | Multiply |
//           | FakeQuantize |                   |  |  |  +----+-----+              +----+-----+
//           +--------------+                   |  |  |       |                         |
//                  |                =====>     |  |  |       |                         |
//                  v                           v  v  v       v                         v
//             +----------+                    +----------------------------------------------+
//             | Multiply | <--- C6            |                 FakeQuantize                 |
//             +----+-----+                    +----------------------------------------------+
//                  |                                                   |
//                  v                                                   v
//

ngraph::pass::FakeQuantizeMulFusion::FakeQuantizeMulFusion() {
  const auto fq_output_low_p = ngraph::pattern::wrap_type<opset4::Constant>();
  const auto fq_output_high_p = ngraph::pattern::wrap_type<opset4::Constant>();

  const auto fq_node_p = ngraph::pattern::wrap_type<opset4::FakeQuantize>(
      {ngraph::pattern::wrap_type<opset4::Constant>(),
       ngraph::pattern::wrap_type<opset4::Constant>(),
       ngraph::pattern::wrap_type<opset4::Constant>(),
       fq_output_low_p,
       fq_output_high_p},
      pattern::consumers_count(1));

  const auto mul_constant_p = ngraph::pattern::wrap_type<opset4::Constant>();
  const auto mul_node_p = ngraph::pattern::wrap_type<opset4::Multiply>(
      {fq_node_p, mul_constant_p}, pattern::consumers_count(1));

  ngraph::matcher_pass_callback callback = [=](pattern::Matcher &m) {
    auto pattern_map = m.get_pattern_value_map();

    const auto fq_node = std::dynamic_pointer_cast<ngraph::opset4::FakeQuantize>(
        pattern_map[fq_node_p].get_node_shared_ptr());
    const auto output_low_const = pattern_map[fq_output_low_p].get_node_shared_ptr();
    const auto output_high_const = pattern_map[fq_output_high_p].get_node_shared_ptr();
    const auto mul_node = pattern_map[mul_node_p].get_node_shared_ptr();
    const auto mul_constant = pattern_map[mul_constant_p].get_node_shared_ptr();

    auto fq_data_shape = fq_node->input_value(0).get_partial_shape();
    const bool mul_constant_matches_fq_data = PartialShape::broadcast_merge_into(
        fq_data_shape, mul_constant->get_output_partial_shape(0), fq_node->get_auto_broadcast());

    if (!mul_constant_matches_fq_data) {
      return false;
    }

    try {
      // create two copies of the original Mul node and use them to multiply the FQ out_* constants
      // the following 2 lines might throw a validation error if the mul_constant's shape
      // does not match the shape of of the output_*_const - in this case we can't modify the graph
      const auto multiplied_out_low = mul_node->clone_with_new_inputs({output_low_const, mul_constant});
      const auto multiplied_out_high = mul_node->clone_with_new_inputs({output_high_const, mul_constant});

      // attach the new Mul nodes to the third and fourth input of the FQ node
      const auto fq_out_low_input = fq_node->input(3);
      fq_out_low_input.replace_source_output(multiplied_out_low->output(0));
      const auto fq_out_high_input = fq_node->input(4);
      fq_out_high_input.replace_source_output(multiplied_out_high->output(0));

      // attach the output of FQ node to the output of the original Mul node
      // (this removes the original Mul node from the graph)
      const auto mul_node_out = mul_node->output(0);
      const auto fq_node_out = fq_node->output(0);
      const auto mul_node_target_input = *(mul_node_out.get_target_inputs().begin());
      mul_node_target_input.replace_source_output(fq_node_out);
    } catch(const ngraph::NodeValidationFailure&) {
      return false;
    }

    return true;
  };

  auto m = std::make_shared<ngraph::pattern::Matcher>(mul_node_p,
                                                      "FakeQuantizeMulFusion");
  this->register_matcher(m, callback);
}
