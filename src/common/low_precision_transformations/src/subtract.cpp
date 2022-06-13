// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/subtract.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

SubtractTransformation::SubtractTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(SubtractTransformation);
    auto convert = pattern::wrap_type<opset1::Convert>();
    auto multiply = pattern::wrap_type<opset1::Multiply>();
    auto subParent = std::make_shared<pattern::op::Or>(OutputVector{ convert, multiply });
    auto subtract = pattern::wrap_type<opset1::Subtract>({ subParent, pattern::wrap_type<opset1::Constant>() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        MATCHER_SCOPE_ENABLE(SubtractTransformation);
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(subtract, matcher_name);
    this->register_matcher(m, callback);
}

bool SubtractTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    std::shared_ptr<opset1::Subtract> subtract = ov::as_type_ptr<opset1::Subtract>(m.get_match_root());
    if (!canBeTransformed(context, subtract)) {
        return false;
    }

    const ngraph::element::Type originalPrecision = subtract->get_output_element_type(0);

    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(subtract, defaultPrecisions);
    if (dequantization.multiply != nullptr) {
        // before: Y = X * SC - SH, after:  Y = (X - SH') * SC
        //    X * SC - SH = X * SC - SH' * SC
        //    SH' = SH / SC
        std::shared_ptr<opset1::Subtract> newSubtract = ov::as_type_ptr<opset1::Subtract>(subtract->clone_with_new_inputs({
            dequantization.multiply->input_value(0),
            ngraph::pass::low_precision::fold<ngraph::opset1::Divide>(
                subtract->input_value(1),
                dequantization.multiply->input_value(1))
        }));

        std::shared_ptr<Node> newMultiply = dequantization.multiply->clone_with_new_inputs({
            newSubtract,
            dequantization.multiplyConstant
        });

        replace_node(subtract, newMultiply);
        subtract = newSubtract;
    }

    if (dequantization.subtract != nullptr) {
        std::shared_ptr<opset1::Subtract> newSubtract = ov::as_type_ptr<opset1::Subtract>(subtract->clone_with_new_inputs({
            dequantization.subtract->input_value(0),
            fold<ngraph::opset1::Add>(subtract->input_value(1), dequantization.subtractConstant)
        }));

        replace_node(subtract, newSubtract);
        subtract = newSubtract;
    }

    if (dequantization.convert != nullptr) {
        // issue #43088
        // std::shared_ptr<Node> newSubtract = NetworkHelper::optimizeElementwise(subtract);
        subtract->set_output_type(0, originalPrecision, subtract->get_output_partial_shape(0));

        replace_node(subtract, std::make_shared<op::TypeRelaxed<opset1::Subtract>>(
            subtract->input_value(0),
            subtract->input_value(1)));
    }
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
