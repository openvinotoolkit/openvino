// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include <ngraph_ops/quantize_conv_bias_fused.hpp>

#include "convert_quantize_conv_elimination.hpp"

#include "ngraph/op/convert.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/pattern/matcher.hpp"


void ngraph::pass::ConvertElimination::convert_elimination() {
    auto const_node = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto convert = std::make_shared<ngraph::op::Convert>(const_node, element::i32);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto convert_node = std::dynamic_pointer_cast<ngraph::op::Convert>(m.get_match_root());
        if (!convert_node || convert_node->get_convert_element_type() != element::i32) {
            return false;
        }

        auto const_node = std::dynamic_pointer_cast<ngraph::op::Constant> (convert_node->get_argument(0));
        if (!const_node) {
            return false;
        }

        // Check that all Convert node consumers are QuantizedConvolutions
        // TODO: how to check this???
        //  for (const auto & output : m.get_match_root()->get_inputs()) {
        //      auto output_node = output.get_node();
        //      auto is_quantized_conv = std::dynamic_pointer_cast<ngraph::op::QuantizedConvolutionBias>(output_node);
        //      if (!is_quantized_conv) return false;
        //  }

        ngraph::replace_node(m.get_match_root(), std::dynamic_pointer_cast<Node>(const_node));

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(convert, "CPUFusion.ConvertElimination");
    this->add_matcher(m, callback);
}
