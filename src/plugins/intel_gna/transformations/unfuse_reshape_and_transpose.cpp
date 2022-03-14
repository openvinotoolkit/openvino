// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/unfuse_reshape_and_transpose.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations/utils/transformation_helper.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>


using namespace GNAPluginNS;

Unfuse2dto4dReshapeAndTranspose::Unfuse2dto4dReshapeAndTranspose() {
    MATCHER_SCOPE(Unfuse2dto4dReshapeAndTranspose);
    auto is_required_reshape = [](const ngraph::Output<ngraph::Node>& value) {
        auto input_shape = value.get_node_shared_ptr()->get_input_shape(0);
        auto output_shape = value.get_node_shared_ptr()->get_output_shape(0);
        return ((input_shape.size() == 2) && (output_shape.size() == 4) &&
                ((output_shape.at(1) == 1) || (output_shape.at(2)*output_shape.at(3) == 1)));
    };
    const auto reshape = ngraph::pattern::wrap_type<ngraph::opset8::Reshape>(is_required_reshape);
    auto fq = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({reshape,
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()},
        consumers_and_rank(1, 4));
    const auto conv = ngraph::pattern::wrap_type<ngraph::opset8::Convolution>({std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{reshape, fq}),
        ngraph::pattern::any_input()});
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto reshape_node = pattern_map.at(reshape).get_node_shared_ptr();
        auto consumers = reshape_node->output(0).get_target_inputs();

        auto N = reshape_node->get_output_shape(0)[0];
        auto C = reshape_node->get_output_shape(0)[1];
        auto H = reshape_node->get_output_shape(0)[2];
        auto W = reshape_node->get_output_shape(0)[3];

        // Create reshape NxW => NxHxWxC (C or HxW is equal to 1)
        auto data = reshape_node->input_value(0);
        auto reshape_nhwc_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, ngraph::Shape{N, H, W, C});
        auto reshape_nhwc = register_new_node<ngraph::opset8::Reshape>(data, reshape_nhwc_const, false);
        reshape_nhwc->set_friendly_name(reshape_node->get_friendly_name() + "/Reshape");

        // Create transpose NxHxWxC => NxCxHxW
        auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 3, 1, 2});
        auto transpose = register_new_node<ngraph::opset8::Transpose>(reshape_nhwc, transpose_const);
        transpose->set_friendly_name(reshape_node->get_friendly_name());

        ngraph::copy_runtime_info(reshape, {reshape_nhwc, transpose});
        for (auto consumer : consumers) {
            consumer.replace_source_output(transpose);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(conv, matcher_name);
    this->register_matcher(m, callback);
}

Unfuse4dto2dReshapeAndTranspose::Unfuse4dto2dReshapeAndTranspose() {
    MATCHER_SCOPE(Unfuse4dto2dReshapeAndTranspose);
    auto is_required_reshape = [](const ngraph::Output<ngraph::Node>& value) {
        auto input_shape = value.get_node_shared_ptr()->get_input_shape(0);
        auto output_shape = value.get_node_shared_ptr()->get_output_shape(0);
        return ((input_shape.size() == 4) && (output_shape.size() == 2) &&
                ((input_shape.at(1) == 1) || (input_shape.at(2)*input_shape.at(3) == 1)));
    };
    // Convolution
    auto conv = ngraph::pattern::wrap_type<ngraph::opset8::Convolution>({ngraph::pattern::any_input(), ngraph::pattern::any_input()},
        consumers_and_rank(1, 4));
    auto fq_conv = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({conv,
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()},
        consumers_and_rank(1, 4));
    // Bias
    auto bias = ngraph::pattern::wrap_type<ngraph::opset8::Add>({conv, ngraph::pattern::any_input()},
        consumers_and_rank(1, 4));
    auto fq_bias = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({bias,
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()},
        consumers_and_rank(1, 4));
    // Max Pooling
    auto max_pool_conv = ngraph::pattern::wrap_type<ngraph::opset7::MaxPool>({conv},
        consumers_and_rank(1, 4));
    auto max_pool_fq_conv = ngraph::pattern::wrap_type<ngraph::opset7::MaxPool>({fq_conv},
        consumers_and_rank(1, 4));
    auto max_pool_bias = ngraph::pattern::wrap_type<ngraph::opset7::MaxPool>({bias},
        consumers_and_rank(1, 4));
    auto max_pool_fq_bias = ngraph::pattern::wrap_type<ngraph::opset7::MaxPool>({fq_bias},
        consumers_and_rank(1, 4));
    // Activation
    auto fq_fq_conv = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({fq_conv,
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()},
        consumers_and_rank(1, 4));
    auto fq_fq_bias = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({fq_bias,
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()},
        consumers_and_rank(1, 4));
    auto act_conv = ngraph::pattern::wrap_type<ngraph::opset8::Relu, ngraph::opset8::Sigmoid,
        ngraph::opset8::Tanh, ngraph::opset8::Abs, ngraph::opset8::Log, ngraph::opset8::Exp,
        ngraph::opset8::Sign, ngraph::opset8::Clamp>({conv},
        consumers_and_rank(1, 4));
    auto act_bias = ngraph::pattern::wrap_type<ngraph::opset8::Relu, ngraph::opset8::Sigmoid,
        ngraph::opset8::Tanh, ngraph::opset8::Abs, ngraph::opset8::Log, ngraph::opset8::Exp,
        ngraph::opset8::Sign, ngraph::opset8::Clamp>({bias},
        consumers_and_rank(1, 4));
    auto act_max_pool_conv = ngraph::pattern::wrap_type<ngraph::opset8::Relu, ngraph::opset8::Sigmoid,
        ngraph::opset8::Tanh, ngraph::opset8::Abs, ngraph::opset8::Log, ngraph::opset8::Exp,
        ngraph::opset8::Sign, ngraph::opset8::Clamp>({max_pool_conv},
        consumers_and_rank(1, 4));
    auto act_max_pool_bias = ngraph::pattern::wrap_type<ngraph::opset8::Relu, ngraph::opset8::Sigmoid,
        ngraph::opset8::Tanh, ngraph::opset8::Abs, ngraph::opset8::Log, ngraph::opset8::Exp,
        ngraph::opset8::Sign, ngraph::opset8::Clamp>({max_pool_bias},
        consumers_and_rank(1, 4));
    auto act_fq_fq_conv = ngraph::pattern::wrap_type<ngraph::opset8::Relu, ngraph::opset8::Sigmoid,
        ngraph::opset8::Tanh, ngraph::opset8::Abs, ngraph::opset8::Log, ngraph::opset8::Exp,
        ngraph::opset8::Sign, ngraph::opset8::Clamp>({fq_fq_conv},
        consumers_and_rank(1, 4));
    auto act_fq_fq_bias = ngraph::pattern::wrap_type<ngraph::opset8::Relu, ngraph::opset8::Sigmoid,
        ngraph::opset8::Tanh, ngraph::opset8::Abs, ngraph::opset8::Log, ngraph::opset8::Exp,
        ngraph::opset8::Sign, ngraph::opset8::Clamp>({fq_fq_bias},
        consumers_and_rank(1, 4));
    auto fq_max_pool_fq_conv = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({max_pool_fq_conv,
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()},
        consumers_and_rank(1, 4));
    auto act_fq_max_pool_fq_conv = ngraph::pattern::wrap_type<ngraph::opset8::Relu, ngraph::opset8::Sigmoid,
        ngraph::opset8::Tanh, ngraph::opset8::Abs, ngraph::opset8::Log, ngraph::opset8::Exp,
        ngraph::opset8::Sign, ngraph::opset8::Clamp>({fq_max_pool_fq_conv},
        consumers_and_rank(1, 4));
    auto fq_max_pool_fq_bias = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({max_pool_fq_bias,
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()},
        consumers_and_rank(1, 4));
    auto act_fq_max_pool_fq_bias = ngraph::pattern::wrap_type<ngraph::opset8::Relu, ngraph::opset8::Sigmoid,
        ngraph::opset8::Tanh, ngraph::opset8::Abs, ngraph::opset8::Log, ngraph::opset8::Exp,
        ngraph::opset8::Sign, ngraph::opset8::Clamp>({fq_max_pool_fq_bias},
        consumers_and_rank(1, 4));
    auto fq_act_fq_fq_conv = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({act_fq_fq_conv,
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()},
        consumers_and_rank(1, 4));
    auto fq_act_fq_fq_bias = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({act_fq_fq_bias,
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()},
        consumers_and_rank(1, 4));
    auto fq_act_fq_max_pool_fq_conv = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({act_fq_max_pool_fq_conv,
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()},
        consumers_and_rank(1, 4));
    auto fq_act_fq_max_pool_fq_bias = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({act_fq_max_pool_fq_bias,
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()},
        consumers_and_rank(1, 4));
    auto root_reshape =
        std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{conv, bias, max_pool_conv, max_pool_fq_conv, max_pool_bias, max_pool_fq_bias,
            fq_conv, fq_bias, act_conv, act_bias, act_max_pool_conv, act_max_pool_bias,
            fq_act_fq_fq_conv, fq_act_fq_fq_bias, fq_act_fq_max_pool_fq_conv, fq_act_fq_max_pool_fq_bias});
    const auto reshape = ngraph::pattern::wrap_type<ngraph::opset8::Reshape>({root_reshape, ngraph::pattern::any_input()}, is_required_reshape);
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto reshape_node = pattern_map.at(reshape).get_node_shared_ptr();
        auto consumers = reshape_node->output(0).get_target_inputs();

        auto N = reshape_node->get_input_shape(0)[0];
        auto W = reshape_node->get_input_shape(0)[1]*reshape_node->get_input_shape(0)[2]*reshape_node->get_input_shape(0)[3];

        // Create transpose NxCxHxW => NxHxWxC
        auto data = reshape_node->input_value(0);
        auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1});
        auto transpose = register_new_node<ngraph::opset8::Transpose>(data, transpose_const);
        transpose->set_friendly_name(reshape_node->get_friendly_name()  + "/Transpose");

        // Create reshape NxHxWxC => NxW (C or HxW is equal to 1)
        auto reshape_nw_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, ngraph::Shape{N, W});
        auto reshape_nw = register_new_node<ngraph::opset8::Reshape>(transpose, reshape_nw_const, false);
        reshape_nw->set_friendly_name(reshape_node->get_friendly_name());

        ngraph::copy_runtime_info(reshape_node, {transpose, reshape_nw});
        for (auto consumer : consumers) {
            consumer.replace_source_output(reshape_nw);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape, matcher_name);
    this->register_matcher(m, callback);
}
