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
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ov {
namespace pass {
namespace low_precision {

MultiplyTransformation::MultiplyTransformation(const Params& params) : EltwiseBaseTransformation(params) {
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

    NetworkHelper::normalizeDequantization(NetworkHelper::getDequantization(multiply, defaultPrecisions, 0));
    NetworkHelper::normalizeDequantization(NetworkHelper::getDequantization(multiply, defaultPrecisions, 1));

    multiply = NetworkHelper::separateInStandaloneBranch(multiply, defaultPrecisions);
    auto newMultiply = multiply;

    auto fold_fake_quantizes = [](std::shared_ptr<Node>& multiply, const size_t index) {
        auto fakeQuantizeOnWeights = ov::as_type_ptr<ov::opset1::FakeQuantize>(multiply->get_input_node_shared_ptr(index));
        if (fakeQuantizeOnWeights != nullptr) {
            auto result = NetworkHelper::fold_fake_quantize(fakeQuantizeOnWeights);
            if (ov::is_type<ov::opset1::Constant>(result)) {
                replace_node(fakeQuantizeOnWeights, result);
            }
        }
    };

    fold_fake_quantizes(multiply, 0ul);
    fold_fake_quantizes(multiply, 1ul);

    const auto dequantization1 = NetworkHelper::foldDequantization(multiply, 0, defaultPrecisions);
    const auto dequantization2 = NetworkHelper::foldDequantization(multiply, 1, defaultPrecisions);
    if ((dequantization1.multiplyConstant == nullptr) && (dequantization2.multiplyConstant == nullptr)) {
        return false;
    }

    // before: Y = (SC1 * (X1 - SH1)) * (SC2 * (X2 - SH2))
    // after : Y = (SC1' * X1` * X2`) , where :
    //         X1` = X1 - SH1
    //         X2` = X2 - SH2
    //         SC1' = SC1 * SC2

    if ((dequantization1.empty() && (ov::is_type<ov::opset1::Constant>(dequantization1.data.get_node()))) ||
        (dequantization2.empty() && (ov::is_type<ov::opset1::Constant>(dequantization2.data.get_node())))) {
        // one input is constant
        auto new_scales_values = fold<ov::opset1::Multiply>(
            dequantization1.empty() ? dequantization1.data : dequantization1.multiplyConstant,
            dequantization2.empty() ? dequantization2.data : dequantization2.multiplyConstant);

        if (!ov::is_type<ov::opset1::Constant>(new_scales_values)) {
            return false;
        }

        const Output<Node> in1 = dequantization1.empty() ?
            new_scales_values :
            dequantization1.subtract == nullptr ?
                dequantization1.data :
                NetworkHelper::optimizeSubtract(dequantization1.subtract);

        const Output<Node> in2 = dequantization2.empty() ?
            new_scales_values :
            dequantization2.subtract == nullptr ?
                dequantization2.data :
                NetworkHelper::optimizeSubtract(dequantization2.subtract);

        auto const new_multiply = (in1.get_element_type() == multiply->get_output_element_type(0)) &&
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

    auto new_scales_values = fold<ov::opset1::Multiply>(dequantization1.multiplyConstant, dequantization2.multiplyConstant);
    if (!ov::is_type<ov::opset1::Constant>(new_scales_values)) {
        return false;
    }

    const Output<Node> in1 = dequantization1.subtract == nullptr ?
        dequantization1.data :
        NetworkHelper::optimizeSubtract(dequantization1.subtract);

    const Output<Node> in2 = dequantization2.subtract == nullptr ?
        dequantization2.data :
        NetworkHelper::optimizeSubtract(dequantization2.subtract);

    // in1 & in2 can have different input types
    const auto new_multiply = (in1.get_element_type() == deqPrecision) &&
                              (in2.get_element_type() == deqPrecision) ?
        std::make_shared<ov::opset1::Multiply>(in1, in2) :
        std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
            std::vector<ov::element::Type>{ deqPrecision, deqPrecision },
            std::vector<ov::element::Type>{ deqPrecision },
            ov::op::TemporaryReplaceOutputType(in1, deqPrecision).get(),
            ov::op::TemporaryReplaceOutputType(in2, deqPrecision).get());

    NetworkHelper::copyInfo(multiply, newMultiply);

    auto new_scales = (new_multiply->get_output_element_type(0) == multiply->get_output_element_type(0)) &&
                      (new_scales_values->get_output_element_type(0) == multiply->get_output_element_type(0)) ?
        std::make_shared<ov::opset1::Multiply>(new_multiply, new_scales_values) :
        std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
            ov::opset1::Multiply(new_multiply, new_scales_values),
            multiply->get_output_element_type(0));

    replace_node(multiply, new_scales);
    updateOutput(context, new_scales, multiply);

    return true;
}

bool MultiplyTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    FakeQuantizeDequantization dequantization1 = pass::low_precision::NetworkHelper::getDequantization(layer, defaultPrecisions, 0ul);
    FakeQuantizeDequantization dequantization2 = pass::low_precision::NetworkHelper::getDequantization(layer, defaultPrecisions, 1ul);

    if (dequantization1.data.get_node() == nullptr || dequantization2.data.get_node() == nullptr) {
        return false;
    }

    const bool nonConstantData = !ov::is_type<ov::opset1::Constant>(dequantization1.data.get_node_shared_ptr()) &&
                                 !ov::is_type<ov::opset1::Constant>(dequantization2.data.get_node_shared_ptr());

    if (((dequantization1.empty() || dequantization2.empty()) && nonConstantData)) {
        return false;
    }

    return EltwiseBaseTransformation::canBeTransformed(context, layer);
}

} // namespace low_precision
} // namespace pass
} // namespace ov
