// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/multiply.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cassert>

#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/rt_info/disable_cleanup_attribute.hpp"
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ov {
namespace pass {
namespace low_precision {

MultiplyTransformation::MultiplyTransformation(const Params& params) :
    WeightableLayerTransformation(params, CanBeTransformedParams(false, false, false, true)) {
    MATCHER_SCOPE(MultiplyTransformation);
    auto matcher = pattern::wrap_type<ov::opset1::Multiply>();

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool MultiplyTransformation::transform(TransformationContext& context, ov::pass::pattern::Matcher& m) {
    auto multiply = m.get_match_root();
    if (!canBeTransformed(context, multiply)) {
        return false;
    }

    multiply = NetworkHelper::separateInStandaloneBranch(multiply, defaultPrecisions);
    decomposeFakeQuantizeForWeightsPath(multiply);

    const auto dequantization1 = NetworkHelper::getDequantization(multiply, defaultPrecisions, 0);
    const auto dequantization2 = NetworkHelper::getDequantization(multiply, defaultPrecisions, 1);

    if ((dequantization1.multiplyConstant == nullptr) && (dequantization2.multiplyConstant == nullptr)) {
        return false;
    }

    // before: y = (deq_scales1 * (x1 - zero_point1)) * (deq_scales2 * (x2 - zero_point2))
    // after : y = deq_scales1 * deq_scales2 * (x1 - zero_point1) * (x2 - zero_point2)

    auto new_scales_values = fold<ov::opset1::Multiply>(
        dequantization1.empty() ? dequantization1.data : dequantization1.multiplyConstant,
        dequantization2.empty() ? dequantization2.data : dequantization2.multiplyConstant);

    if (!ov::is_type<ov::opset1::Constant>(new_scales_values)) {
        return false;
    }

    const auto init_input = [&new_scales_values](const FakeQuantizeDequantization& dequantization) -> Output<Node> {
        if (dequantization.empty()) {
            return new_scales_values;
        }

        if (dequantization.subtract == nullptr) {
            return dequantization.data;
        }

        const auto subtract = NetworkHelper::optimizeSubtract(dequantization.subtract);
        if (subtract != nullptr) {
            DisableCleanupAttribute::create(subtract);
        }

        return subtract == nullptr ? dequantization.data : subtract;
    };

    if ((dequantization1.empty() && (ov::is_type<ov::opset1::Constant>(dequantization1.data.get_node()))) ||
        (dequantization2.empty() && (ov::is_type<ov::opset1::Constant>(dequantization2.data.get_node())))) {
        // one input is constant
        const Output<Node> in1 = init_input(dequantization1);
        const Output<Node> in2 = init_input(dequantization2);

        const auto new_multiply = (in1.get_element_type() == multiply->get_output_element_type(0)) &&
                                  (in2.get_element_type() == multiply->get_output_element_type(0)) ?
            std::make_shared<ov::opset1::Multiply>(in1, in2) :
            std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
                std::vector<ov::element::Type>{ deqPrecision, deqPrecision },
                std::vector<ov::element::Type>{ multiply->get_output_element_type(0) },
                ov::op::TemporaryReplaceOutputType(in1, deqPrecision).get(),
                ov::op::TemporaryReplaceOutputType(in2, deqPrecision).get());

        replace_node(multiply, new_multiply);
        updateOutput(context, new_multiply, multiply);

        return true;
    }

    Output<Node> in1 = init_input(dequantization1);
    Output<Node> in2 = init_input(dequantization2);

    // in1 & in2 can have different input types
    const auto new_multiply = (in1.get_element_type() == deqPrecision) &&
                              (in2.get_element_type() == deqPrecision) ?
        std::make_shared<ov::opset1::Multiply>(in1, in2) :
        std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
            std::vector<ov::element::Type>{ deqPrecision, deqPrecision },
            std::vector<ov::element::Type>{ deqPrecision },
            ov::op::TemporaryReplaceOutputType(in1, deqPrecision).get(),
            ov::op::TemporaryReplaceOutputType(in2, deqPrecision).get());

    DisableCleanupAttribute::create(new_multiply);

    auto new_scales = (new_multiply->get_output_element_type(0) == multiply->get_output_element_type(0)) &&
                      (new_scales_values->get_output_element_type(0) == multiply->get_output_element_type(0)) ?
        std::make_shared<ov::opset1::Multiply>(new_multiply, new_scales_values) :
        std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
            ov::opset1::Multiply(new_multiply, new_scales_values),
            multiply->get_output_element_type(0));

    replace_node(multiply, new_scales);
    const auto was_updated = updateOutput(context, new_scales, multiply);
    NetworkHelper::copyInfo(multiply, new_multiply, !was_updated);

    return true;
}

size_t MultiplyTransformation::getInputChannels(const std::shared_ptr<ov::Node> op) const {
    const auto channels = op->get_input_partial_shape(1)[1];
    assert(channels.is_static());
    return channels.get_length();
}

} // namespace low_precision
} // namespace pass
} // namespace ov
