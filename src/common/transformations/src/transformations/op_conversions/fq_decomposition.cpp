// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/fq_decomposition.hpp"

#include <ngraph/builder/autobroadcast.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <numeric>

#include "itt.hpp"

namespace {

bool isValidRangesInputs(const std::shared_ptr<ngraph::opset1::FakeQuantize>& fq) {
    auto il = fq->input_value(1);
    auto ih = fq->input_value(2);
    auto greater_equal = std::make_shared<ngraph::opset1::GreaterEqual>(il, ih);

    ngraph::OutputVector result(1);
    if (!greater_equal->constant_fold(result, greater_equal->input_values()))
        return false;

    auto res_node = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(result[0].get_node_shared_ptr());

    const std::vector<bool> comp_result = res_node->cast_vector<bool>();

    return !std::any_of(comp_result.begin(), comp_result.end(), [](const bool value) {
        return value;
    });
}

}  // namespace

ngraph::pass::FakeQuantizeDecomposition::FakeQuantizeDecomposition() {
    MATCHER_SCOPE(FakeQuantizeDecomposition);
    auto data = ngraph::pattern::any_input();
    auto il = ngraph::pattern::wrap_type<opset1::Constant>();
    auto ih = ngraph::pattern::wrap_type<opset1::Constant>();
    auto ol = ngraph::pattern::wrap_type<opset1::Constant>();
    auto oh = ngraph::pattern::wrap_type<opset1::Constant>();
    auto fake_quantize = ngraph::pattern::wrap_type<ngraph::opset1::FakeQuantize>({data, il, ih, ol, oh});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        const auto fake_quantize_node = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(
            pattern_to_output.at(fake_quantize).get_node_shared_ptr());

        if (fake_quantize_node == nullptr || transformation_callback(fake_quantize_node) ||
            !isValidRangesInputs(fake_quantize_node)) {
            return false;
        }

        Output<Node> data{fake_quantize_node->input_value(0)};
        const Output<Node> input_low{fake_quantize_node->input_value(1)};
        const Output<Node> input_high{fake_quantize_node->input_value(2)};
        const Output<Node> output_low{fake_quantize_node->input_value(3)};
        const Output<Node> output_high{fake_quantize_node->input_value(4)};
        auto input_type = data.get_element_type();

        ngraph::NodeVector decomp_ops;
        if (input_type != input_low.get_element_type()) {
            input_type = input_low.get_element_type();
            data = std::make_shared<ngraph::opset1::Convert>(data, input_type);
            decomp_ops.push_back(data.get_node_shared_ptr());
        }

        // if we set input_low or input_high in formula we got output = output_low and output = output_high respectively
        // so we just clamp x
        const auto max = std::make_shared<ngraph::opset1::Maximum>(data, input_low);
        const auto min = std::make_shared<ngraph::opset1::Minimum>(max, input_high);
        decomp_ops.push_back(max);
        decomp_ops.push_back(min);

        // (levels-1)
        const auto levels_minus_one =
            std::make_shared<ngraph::opset1::Constant>(input_type, Shape{}, fake_quantize_node->get_levels() - 1);
        decomp_ops.push_back(levels_minus_one);
        // (input_high - input_low)
        const auto subInHighLow = std::make_shared<ngraph::opset1::Subtract>(input_high, input_low);
        // (levels-1) / (input_high - input_low)
        const auto isc = std::make_shared<ngraph::opset1::Divide>(levels_minus_one, subInHighLow);
        // input_low * (levels-1) / (input_high - input_low)
        const auto ish = std::make_shared<ngraph::opset1::Multiply>(input_low, isc);
        decomp_ops.push_back(subInHighLow);
        decomp_ops.push_back(isc);
        decomp_ops.push_back(ish);

        // x * (levels-1) / (input_high - input_low)
        const auto after_isc_apply = std::make_shared<ngraph::opset1::Multiply>(min, isc);
        // x * (levels-1) / (input_high - input_low) - input_low * (levels-1) / (input_high - input_low)
        const auto after_ish_apply = std::make_shared<ngraph::opset1::Subtract>(after_isc_apply, ish);
        decomp_ops.push_back(after_isc_apply);
        decomp_ops.push_back(after_ish_apply);

        // round(x * (levels-1) / (input_high - input_low) - input_low * (levels-1) / (input_high - input_low))
        const auto round =
            std::make_shared<ngraph::opset5::Round>(after_ish_apply, ngraph::opset5::Round::RoundMode::HALF_TO_EVEN);
        decomp_ops.push_back(round);

        // (output_high - output_low)
        const auto sub_out_high_low = std::make_shared<ngraph::opset1::Subtract>(output_high, output_low);
        // (output_high - output_low) / (levels-1)
        const auto osc = std::make_shared<ngraph::opset1::Divide>(sub_out_high_low, levels_minus_one);
        decomp_ops.push_back(sub_out_high_low);
        decomp_ops.push_back(osc);

        // round(x * (levels-1) / (input_high - input_low) - input_low * (levels-1) / (input_high - input_low)) *
        // (output_high - output_low) / (levels-1)
        const auto after_osc_apply = std::make_shared<ngraph::opset1::Multiply>(round, osc);
        // round(x * (levels-1) / (input_high - input_low) - input_low * (levels-1) / (input_high - input_low)) *
        // (output_high - output_low) / (levels-1) + output_low
        std::shared_ptr<Node> result = std::make_shared<ngraph::opset1::Add>(after_osc_apply, output_low);
        decomp_ops.push_back(after_osc_apply);
        decomp_ops.push_back(result);

        if (result->get_output_element_type(0) != fake_quantize_node->get_output_element_type(0)) {
            result = std::make_shared<ngraph::opset1::Convert>(result, fake_quantize_node->get_output_element_type(0));
            decomp_ops.push_back(result);
        }

        result->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(fake_quantize_node, decomp_ops);
        ngraph::replace_node(m.get_match_root(), result);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fake_quantize, matcher_name);
    register_matcher(m, callback);
}
