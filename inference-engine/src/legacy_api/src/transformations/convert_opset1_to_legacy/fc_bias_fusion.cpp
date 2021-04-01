// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/transformations/convert_opset1_to_legacy/fc_bias_fusion.hpp"

#include <memory>
#include <numeric>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::FullyConnectedBiasFusion, "FullyConnectedBiasFusion", 0);

ngraph::pass::FullyConnectedBiasFusion::FullyConnectedBiasFusion() {
    auto m_fc = ngraph::pattern::wrap_type<op::FullyConnected>([](Output<Node> output) {
        return pattern::consumers_count(1)(output) &&
               pattern::has_static_shape()(output);
    });
    auto m_bias = pattern::any_input();
    auto m_add = ngraph::pattern::wrap_type<opset1::Add>({m_fc, m_bias});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher &m) {
        auto & pattern_to_output = m.get_pattern_value_map();

        auto add = pattern_to_output[m_add].get_node_shared_ptr();
        auto bias = pattern_to_output[m_bias].get_node_shared_ptr();
        auto fc = std::dynamic_pointer_cast<op::FullyConnected>(pattern_to_output[m_fc].get_node_shared_ptr());
        if (!fc) {
            return false;
        }

        if (auto bcast = std::dynamic_pointer_cast<opset1::Broadcast>(bias)) {
            bias = bcast->input_value(0).get_node_shared_ptr();
        }

        if (!std::dynamic_pointer_cast<opset1::Constant>(bias)) {
            return false;
        }

        Shape bias_shape(bias->get_shape());
        Shape output_shape(fc->get_shape());
        size_t bias_size = std::accumulate(bias_shape.begin(), bias_shape.end(), size_t{1}, std::multiplies<int64_t>());
        if (bias_shape.empty() || bias_shape.back() != output_shape.back() || bias_shape.back() != bias_size) {
            return false;
        }

        NodeVector new_ops;

        auto new_bias = std::make_shared<opset1::Add>(fc->input(2).get_source_output(), bias);
        new_ops.push_back(new_bias);
        std::shared_ptr<Node> final_bias = new_bias;
        if (new_bias->get_shape().size() >= 2) {
            final_bias = std::make_shared<opset1::Reshape>(final_bias, opset1::Constant::create(element::i64, Shape{1}, {-1}), true);
            new_ops.push_back(final_bias);
        }

        auto new_fc = std::make_shared<op::FullyConnected>(fc->input(0).get_source_output(),
                                                           fc->input(1).get_source_output(),
                                                           final_bias,
                                                           fc->get_shape(),
                                                           fc->get_output_type());
        new_ops.push_back(new_fc);

        new_fc->set_friendly_name(add->get_friendly_name());
        ngraph::copy_runtime_info({fc, add}, new_ops);
        ngraph::replace_node(add, new_fc);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_add, "FullyConnectedBiasFusion");
    this->register_matcher(m, callback);
}
