// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/disable_convert_constant_folding_on_const_path.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <queue>
#include <transformations/rt_info/disable_constant_folding.hpp>
#include <vector>

using namespace ngraph;

ngraph::pass::DisableConvertConstantFoldingOnConstPath::DisableConvertConstantFoldingOnConstPath(
    const element::TypeVector& inputPrecisions) {
    auto matcherData = ngraph::pattern::any_input();
    auto matcherConvert = ngraph::pattern::wrap_type<opset3::Convert>({matcherData}, pattern::consumers_count(1));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) -> bool {
        const auto& opsMap = m.get_pattern_value_map();
        const auto convert = opsMap.at(matcherConvert).get_node_shared_ptr();

        // validation by Convert operation input precisions
        if (!inputPrecisions.empty()) {
            const ngraph::element::Type inputPrecision = convert->input(0).get_element_type();
            if (std::find(inputPrecisions.begin(), inputPrecisions.end(), inputPrecision) == inputPrecisions.end()) {
                return false;
            }
        }

        // Constant subgraph has not be folded if Convert is part of dequantization operations:
        //
        //   Constant                             Constant
        //      |                                    |
        //   Convert  Constant           OR       Convert  Constant
        //       \     /                             \      /
        //       Subtract   Constant                 Multiply
        //           \      /
        //           Multiply
        //
        auto parent = convert->get_input_node_ptr(0);
        auto target_inputs = convert->output(0).get_target_inputs();
        if (target_inputs.empty()) {
            return false;
        }
        auto child = target_inputs.begin()->get_node();
        if (ov::is_type<ngraph::opset1::Constant>(parent) &&
            (ov::is_type<ngraph::opset1::Subtract>(child) || ov::is_type<ngraph::opset1::Multiply>(child))) {
            ov::disable_constant_folding(convert);
            return true;
        }

        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcherConvert, "DisableConvertConstantFoldingOnConstPath");
    this->register_matcher(m, callback);
}
