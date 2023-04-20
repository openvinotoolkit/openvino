// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_bias_flatten.hpp"
#include "transformations/cpu_opset/common/op/fully_connected.hpp"
#include <numeric>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "transformations/utils/utils.hpp"

#include "itt.hpp"
namespace {
bool flatten_bias(std::shared_ptr<ngraph::Node> bias, const ov::PartialShape& fc_output_shape) {
    const ngraph::Shape& bias_shape(bias->get_shape());
    const auto bias_size = ngraph::shape_size(bias_shape);
    const auto rank = fc_output_shape.rank().get_length();
    //output OC dimension must be static dimension.
    if (rank == 0 || fc_output_shape[rank - 1].is_dynamic())
        return false;
    if (bias_shape.empty() || bias_shape.back() != fc_output_shape[rank - 1].get_length() || bias_shape.back() != bias_size)
        return false;
    if (bias_shape.size() == 1)
        return false;

    std::shared_ptr<ngraph::Node> new_bias = bias;

    auto reshape_const = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { -1 });
    new_bias = ov::op::util::make_try_fold<ngraph::opset1::Reshape>(new_bias, reshape_const, true);
    new_bias->set_friendly_name(bias->get_friendly_name());
    ngraph::copy_runtime_info({bias}, new_bias);
    ngraph::replace_node(bias, new_bias);
    return true;
}
} // namespace

//Check the pattern of "FC + BIAS".
ov::intel_cpu::NonQuantizedFullyConnectedBiasFlatten::NonQuantizedFullyConnectedBiasFlatten() {
    MATCHER_SCOPE(FullyConnectedBiasFusion);
    auto input = ngraph::pattern::any_input();
    auto weights = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto m_fc = ngraph::pattern::wrap_type<ov::intel_cpu::FullyConnectedNode>({ input, weights }, [](ngraph::Output<ngraph::Node> output) {
        return ngraph::pattern::consumers_count(1)(output) && ngraph::pattern::has_static_rank()(output);
    });
    auto m_bias = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto m_add = ngraph::pattern::wrap_type<ngraph::opset1::Add>({m_fc, m_bias});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        auto& pattern_to_output = m.get_pattern_value_map();

        auto add = pattern_to_output[m_add].get_node_shared_ptr();
        auto bias = pattern_to_output[m_bias].get_node_shared_ptr();
        auto fc = std::dynamic_pointer_cast<ov::intel_cpu::FullyConnectedNode>(pattern_to_output[m_fc].get_node_shared_ptr());
        if (!fc || transformation_callback(fc)) {
            return false;
        }

        if (!std::dynamic_pointer_cast<ngraph::opset1::Constant>(bias)) {
            return false;
        }

        const ngraph::PartialShape& output_shape(fc->get_output_partial_shape(0));
        return flatten_bias(bias, output_shape);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_add, matcher_name);
    this->register_matcher(m, callback);
}

//CPU plugin would config LPT  not to propogate dequantization scale over bias to follow ONEDNN 3.x scheme.
//Check the pattern of "FC + DQ + BIAS".
ov::intel_cpu::QuantizedFullyConnectedBiasFlatten::QuantizedFullyConnectedBiasFlatten() {
    MATCHER_SCOPE(FullyConnectedBiasFusion);
    auto input = ngraph::pattern::any_input();
    auto weights = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto m_fc = ngraph::pattern::wrap_type<ov::intel_cpu::FullyConnectedNode>({ input, weights }, [](ngraph::Output<ngraph::Node> output) {
        return ngraph::pattern::consumers_count(1)(output) && ngraph::pattern::has_static_rank()(output);
    });

    auto m_scale = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto m_mul = ngraph::pattern::wrap_type<ngraph::opset1::Multiply>({m_fc, m_scale});

    auto m_bias = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto m_add = ngraph::pattern::wrap_type<ngraph::opset1::Add>({m_mul, m_bias});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto mul = pattern_to_output[m_mul].get_node_shared_ptr();
        auto scale = pattern_to_output[m_scale].get_node_shared_ptr();
        auto add = pattern_to_output[m_add].get_node_shared_ptr();
        auto bias = pattern_to_output[m_bias].get_node_shared_ptr();
        auto fc = std::dynamic_pointer_cast<ov::intel_cpu::FullyConnectedNode>(pattern_to_output[m_fc].get_node_shared_ptr());
        if (!fc || transformation_callback(fc)) {
            return false;
        }

        if (!std::dynamic_pointer_cast<ngraph::opset1::Constant>(bias)) {
            return false;
        }

        const ngraph::PartialShape& output_shape(fc->get_output_partial_shape(0));

        ngraph::Shape bias_shape(bias->get_shape());
        //Dupliated check of MarkupBias RTInfo
        const bool per_channel = std::count_if(bias_shape.begin(), bias_shape.end(), [](size_t x) { return x > 1; }) == 1;
        if (ov::shape_size(bias_shape) != 1 && !per_channel)
            return false;
        return flatten_bias(bias, output_shape);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_add, matcher_name);
    this->register_matcher(m, callback);
}