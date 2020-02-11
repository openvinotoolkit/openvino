// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>

#include <ngraph_ops/fully_connected.hpp>
#include <ngraph/builder/make_constant.hpp>
#include <ngraph/graph_util.hpp>
#include <ngraph/op/add.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/label.hpp>
#include <ngraph/pattern/op/skip.hpp>
#include <ngraph/util.hpp>
#include <ngraph/ngraph.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class FullyConnectedBiasFusion;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::FullyConnectedBiasFusion : public ngraph::pass::GraphRewrite {
public:
    FullyConnectedBiasFusion() : GraphRewrite() {
        construct_fcbias();
    }

private:
    void construct_fcbias() {
        Shape shape_w{2, 4};
        Shape shape_x{2, 4};
        Shape shape_b{2, 2};
        auto input = std::make_shared<pattern::op::Label>(element::f32, shape_w);
        auto weights = std::make_shared<pattern::op::Label>(element::f32, shape_x);
        auto fc_bias = std::make_shared<pattern::op::Label>(element::f32, shape_b);
        auto bias = std::make_shared<pattern::op::Label>(element::f32, shape_b);

        auto fc = std::make_shared<ngraph::op::FullyConnected>(input, weights, fc_bias);
        auto add = std::make_shared<ngraph::op::v1::Add>(fc, bias);

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher &m) {
            auto add = m.get_match_root();
            auto add_input_0 = add->input(0).get_source_output().get_node_shared_ptr();
            auto add_input_1 = add->input(1).get_source_output().get_node_shared_ptr();

            auto m_fc = std::dynamic_pointer_cast<ngraph::op::FullyConnected>(add_input_0);
            auto m_bias = add_input_1;

            if (m_fc == nullptr) {
                m_fc = std::dynamic_pointer_cast<ngraph::op::FullyConnected>(add_input_1);
                m_bias = add_input_0;
            }

            if (auto bcast_m = std::dynamic_pointer_cast<ngraph::op::v1::Broadcast>(m_bias)) {
                m_bias = bcast_m->input(0).get_source_output().get_node_shared_ptr();
            }

            if (!std::dynamic_pointer_cast<ngraph::op::Constant>(m_bias)) {
                return false;
            }
            Shape bias_shape(m_bias->get_shape());

            if (m_fc->output(0).get_target_inputs().size() != 1) {
                return false;
            }

            Shape output_shape(m_fc->get_shape());
            size_t bias_size = std::accumulate(bias_shape.begin(), bias_shape.end(), 1, std::multiplies<size_t>());
            if (bias_shape.empty() || bias_shape.back() != output_shape[1] || bias_shape.back() != bias_size) {
                return false;
            }

            auto new_bias = std::make_shared<op::v1::Add>(m_fc->input(2).get_source_output(), m_bias);
            std::shared_ptr<ngraph::Node> final_bias = new_bias;
            if (new_bias->get_shape().size() == 2 && new_bias->get_shape()[0] == 1) {
                final_bias = std::make_shared<ngraph::op::Squeeze>(final_bias, ngraph::op::Constant::create(element::i64, Shape{1}, {0}));
            }
            auto new_fc = std::make_shared<op::FullyConnected>(m_fc->input(0).get_source_output(),
                                                               m_fc->input(1).get_source_output(),
                                                               final_bias);
            new_fc->set_friendly_name(add->get_friendly_name());
            ngraph::replace_node(add, new_fc);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(add, "FullyConnectedBiasFusion");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
