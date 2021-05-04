// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/reorder_activation_and_pooling.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <gna_plugin_log.hpp>

using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(ReorderActivationAndPooling, "ReorderActivationAndPooling", 0);

ReorderActivationAndPooling::ReorderActivationAndPooling() {
    auto conv = ngraph::pattern::wrap_type<ngraph::opset1::Convolution>({ngraph::pattern::any_input(),
                                                                         ngraph::pattern::any_input()});
    auto add = ngraph::pattern::wrap_type<ngraph::opset1::Add>({conv, ngraph::pattern::any_input()});
    auto il = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    auto ih = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    auto ol = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    auto oh = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    auto fq1 = ngraph::pattern::wrap_type<ngraph::opset1::FakeQuantize>({conv, il, ih, ol, oh});
    auto fq2 = ngraph::pattern::wrap_type<ngraph::opset1::FakeQuantize>({add, il, ih, ol, oh});
    auto act1 = ngraph::pattern::wrap_type<ngraph::opset1::Relu, ngraph::opset1::Sigmoid,
            ngraph::opset1::Tanh, ngraph::opset1::Abs, ngraph::opset1::Log, ngraph::opset1::Exp,
            ngraph::opset1::Sign, ngraph::opset1::Clamp>({conv});
    auto act2 = ngraph::pattern::wrap_type<ngraph::opset1::Relu, ngraph::opset1::Sigmoid,
            ngraph::opset1::Tanh, ngraph::opset1::Abs, ngraph::opset1::Log, ngraph::opset1::Exp,
            ngraph::opset1::Sign, ngraph::opset1::Clamp>({add});
    auto act = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{fq1, fq2, act1, act2});
    auto pool = ngraph::pattern::wrap_type<ngraph::opset1::MaxPool>({act});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        auto& pattern_map = m.get_pattern_value_map();
        auto pool_node = pattern_map.at(pool).get_node_shared_ptr();
        auto pool = std::dynamic_pointer_cast<ngraph::opset1::MaxPool>(pool_node);
        IE_ASSERT(pool != nullptr);
        auto kernel_shape = pool->get_kernel();
        if (kernel_shape.size() > 1 && kernel_shape[0] > 1 && kernel_shape[1] > 1) {
            return false;
        }

        auto act = pool_node->input_value(0).get_node_shared_ptr();
        IE_ASSERT(act != nullptr);

        gnalog() << "Reorder " << pool_node->get_friendly_name() << " and  " << act->get_friendly_name() << "\n";

        auto node_before_act = act->input_value(0).get_node_shared_ptr();
        IE_ASSERT(node_before_act != nullptr);

        auto consumers = node_before_act->output(0).get_target_inputs();
        auto new_pool = std::make_shared<ngraph::opset1::MaxPool>(node_before_act, pool->get_strides(), pool->get_pads_begin(),
                                                                  pool->get_pads_end(), kernel_shape, pool->get_rounding_type(),
                                                                  pool->get_auto_pad());
        for (auto input : consumers) {
            input.replace_source_output(new_pool);
        }

        ngraph::replace_output_update_name(pool_node->output(0), pool_node->input_value(0));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(pool, "ReorderActivationAndPooling");
    this->register_matcher(m, callback);
}
