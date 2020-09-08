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
  const auto fq_data_p = ngraph::pattern::wrap_type<opset4::Constant>();
  const auto fq_input_low_p = ngraph::pattern::wrap_type<opset4::Constant>();
  const auto fq_input_high_p = ngraph::pattern::wrap_type<opset4::Constant>();
  const auto fq_output_low_p = ngraph::pattern::wrap_type<opset4::Constant>();
  const auto fq_output_high_p = ngraph::pattern::wrap_type<opset4::Constant>();

  const auto fq_node = ngraph::pattern::wrap_type<opset4::FakeQuantize>(
      {fq_data_p, fq_input_low_p, fq_input_high_p, fq_output_low_p,
       fq_output_high_p},
      pattern::consumers_count(1));

  const auto mul_constant = ngraph::pattern::wrap_type<opset4::Constant>();
  const auto mul_node = ngraph::pattern::wrap_type<opset4::Multiply>(
      {fq_node, mul_constant}, pattern::consumers_count(1));

  ngraph::matcher_pass_callback callback = [=](pattern::Matcher &m) {
    std::cout << "Pattern found\n";
    return false;
  };

  auto m = std::make_shared<ngraph::pattern::Matcher>(mul_node,
                                                      "FakeQuantizeMulFusion");
  this->register_matcher(m, callback);
}
