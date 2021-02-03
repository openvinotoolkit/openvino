// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/disable_convert_constant_folding_on_const_path.hpp"

#include <memory>
#include <queue>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/variant.hpp>
#include "transformations/rt_info/dequantization_attribute.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::DisableConvertConstantFoldingOnConstPath, "DisableConvertConstantFoldingOnConstPath", 0);

ngraph::pass::DisableConvertConstantFoldingOnConstPath::DisableConvertConstantFoldingOnConstPath(
    const std::vector<ngraph::element::Type>& inputPrecisions) {
    auto matcherData = ngraph::pattern::any_input();
    auto matcherConvert = ngraph::pattern::wrap_type<opset3::Convert>({ matcherData }, pattern::consumers_count(1));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher & m) -> bool {
        const auto& opsMap = m.get_pattern_value_map();
        const auto convert = opsMap.find(matcherConvert)->second.get_node()->shared_from_this();

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
        auto child = convert->output(0).get_target_inputs().begin()->get_node();
        if (is_type<ngraph::opset1::Constant>(parent) &&
            (is_type<ngraph::opset1::Subtract>(child) || is_type<ngraph::opset1::Multiply>(child))) {
            auto& rtInfo = convert->get_rt_info();
            rtInfo["DISABLED_CONSTANT_FOLDING"] = std::make_shared<VariantWrapper<std::string>>("");
            return true;
        }

        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcherConvert, "DisableConvertConstantFoldingOnConstPath");
    this->register_matcher(m, callback);
}
