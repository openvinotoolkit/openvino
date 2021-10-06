// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_bias_fusion.hpp"
#include "op/fully_connected.hpp"
#include <numeric>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::FullyConnectedBiasFusion, "FullyConnectedBiasFusion", 0);

MKLDNNPlugin::FullyConnectedBiasFusion::FullyConnectedBiasFusion() {
    auto m_fc = ngraph::pattern::wrap_type<MKLDNNPlugin::FullyConnectedNode>([](ngraph::Output<ngraph::Node> output) {
        return ngraph::pattern::consumers_count(1)(output) && ngraph::pattern::has_static_shape()(output);
    });
    auto m_bias = ngraph::pattern::any_input();
    auto m_add = ngraph::pattern::wrap_type<ngraph::opset1::Add>({m_fc, m_bias});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        auto & pattern_to_output = m.get_pattern_value_map();

        auto add = pattern_to_output[m_add].get_node_shared_ptr();
        auto bias = pattern_to_output[m_bias].get_node_shared_ptr();
        auto fc = std::dynamic_pointer_cast<MKLDNNPlugin::FullyConnectedNode>(pattern_to_output[m_fc].get_node_shared_ptr());
        if (!fc) {
            return false;
        }

        if (auto bcast = std::dynamic_pointer_cast<ngraph::opset1::Broadcast>(bias)) {
            bias = bcast->input_value(0).get_node_shared_ptr();
        }

        if (!std::dynamic_pointer_cast<ngraph::opset1::Constant>(bias)) {
            return false;
        }

        ngraph::Shape bias_shape(bias->get_shape());
        ngraph::Shape output_shape(fc->get_shape());
        size_t bias_size = std::accumulate(bias_shape.begin(), bias_shape.end(), size_t{1}, std::multiplies<int64_t>());
        if (bias_shape.empty() || bias_shape.back() != output_shape.back() || bias_shape.back() != bias_size) {
            return false;
        }

        ngraph::NodeVector new_ops;

        std::shared_ptr<ngraph::Node> final_bias = bias;
        if (bias->get_shape().size() >= 2) {
            final_bias = std::make_shared<ngraph::opset1::Reshape>(final_bias, ngraph::opset1::Constant::create(ngraph::element::i64,
                                                                                                                ngraph::Shape{1}, {-1}), true);
            new_ops.push_back(final_bias);
        }

        auto new_fc = std::make_shared<MKLDNNPlugin::FullyConnectedNode>(fc->input(0).get_source_output(),
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
