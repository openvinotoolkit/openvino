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
    auto fc = ngraph::pattern::wrap_type<op::FullyConnected>();
    auto add = ngraph::pattern::wrap_type<opset1::Add>({fc, std::make_shared<pattern::op::Label>()});

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher &m) {
        auto add = m.get_match_root();
        auto add_input_0 = add->input(0).get_source_output().get_node_shared_ptr();
        auto add_input_1 = add->input(1).get_source_output().get_node_shared_ptr();

        auto m_fc = std::dynamic_pointer_cast<op::FullyConnected>(add_input_0);
        auto m_bias = add_input_1;

        if (m_fc == nullptr) {
            m_fc = std::dynamic_pointer_cast<op::FullyConnected>(add_input_1);
            if (m_fc == nullptr)
                return false;
            m_bias = add_input_0;
        }

        if (auto bcast_m = std::dynamic_pointer_cast<opset1::Broadcast>(m_bias)) {
            m_bias = bcast_m->input(0).get_source_output().get_node_shared_ptr();
        }

        if (!std::dynamic_pointer_cast<opset1::Constant>(m_bias)) {
            return false;
        }
        Shape bias_shape(m_bias->get_shape());

        if (m_fc->output(0).get_target_inputs().size() != 1) {
            return false;
        }

        Shape output_shape(m_fc->get_shape());
        size_t bias_size = std::accumulate(bias_shape.begin(), bias_shape.end(), size_t{1}, std::multiplies<int64_t>());
        if (bias_shape.empty() || bias_shape.back() != output_shape.back() || bias_shape.back() != bias_size) {
            return false;
        }

        NodeVector new_ops;

        auto new_bias = std::make_shared<opset1::Add>(m_fc->input(2).get_source_output(), m_bias);
        new_ops.push_back(new_bias);
        std::shared_ptr<Node> final_bias = new_bias;
        if (new_bias->get_shape().size() >= 2) {
            final_bias = std::make_shared<opset1::Reshape>(final_bias, opset1::Constant::create(element::i64, Shape{1}, {-1}), true);
            new_ops.push_back(final_bias);
        }

        auto new_fc = std::make_shared<op::FullyConnected>(m_fc->input(0).get_source_output(),
                                                           m_fc->input(1).get_source_output(),
                                                           final_bias,
                                                           m_fc->get_shape(),
                                                           m_fc->get_output_type());
        new_ops.push_back(new_fc);

        new_fc->set_friendly_name(add->get_friendly_name());
        ngraph::copy_runtime_info({m_fc, add}, new_ops);
        ngraph::replace_node(add, new_fc);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, "FullyConnectedBiasFusion");
    this->register_matcher(m, callback);
}
