// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/convert_subtract_constant.hpp"

#include <memory>
#include <vector>

#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

using namespace ov;

// Original (FP16 as example, I8 in constantPrecisions):
//
//   Constant
//     | I8
//   Convert     Constant
//      \ FP16   / FP16
//       Subtract    Constant
//         \ FP16   / FP16
//          Multiply
//
// Result:
//
//   Constant    Constant
//     | I8      | I8
//   Convert     Convert
//      \ FP16   / FP16
//       Subtract    Constant
//         \ FP16   / FP16
//          Multiply
//
ov::pass::low_precision::ConvertSubtractConstant::ConvertSubtractConstant(const std::vector<ov::element::Type>& constantPrecisions) {
    MATCHER_SCOPE(ConvertSubtractConstant);
    auto weightsConstantWrapper = ov::pass::pattern::wrap_type<opset1::Constant>(pattern::consumers_count(1));
    auto weightsConvertWrapper = ov::pass::pattern::wrap_type<opset1::Convert>({ weightsConstantWrapper }, pattern::consumers_count(1));
    auto subtractConstantWrapper = ov::pass::pattern::wrap_type<opset1::Constant>(pattern::consumers_count(1));
    auto subtractWrapper = ov::pass::pattern::wrap_type<opset1::Subtract>({ weightsConvertWrapper, subtractConstantWrapper }, pattern::consumers_count(1));
    auto multiplyConstantWrapper = ov::pass::pattern::wrap_type<opset1::Constant>(pattern::consumers_count(1));
    auto multiplyWrapper = ov::pass::pattern::wrap_type<opset1::Multiply>({ subtractWrapper, multiplyConstantWrapper }, pattern::consumers_count(1));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher & m) -> bool {
        const auto& opsMap = m.get_pattern_value_map();
        const auto weightsConvert = opsMap.at(weightsConvertWrapper).get_node_shared_ptr();
        const auto quantizePrecision = weightsConvert->get_input_element_type(0);
        const auto dequantizationPrecision = weightsConvert->get_output_element_type(0);

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        // validation by Convert operation input precisions
        if (!constantPrecisions.empty()) {
            const ov::element::Type inputPrecision = quantizePrecision;
            if (std::find(constantPrecisions.begin(), constantPrecisions.end(), inputPrecision) == constantPrecisions.end()) {
                return false;
            }
        }

        const auto subtract = opsMap.at(subtractWrapper).get_node_shared_ptr();
        if (!NetworkHelper::checkZeroPoint(subtract)) {
            return false;
        }

        const auto subtractConstant = opsMap.at(subtractConstantWrapper).get_node_shared_ptr();
        auto resultSubtractConstant = NetworkHelper::round(subtractConstant, quantizePrecision);
        if (NetworkHelper::isScalarLike(resultSubtractConstant)) {
            resultSubtractConstant = NetworkHelper::toScalar(resultSubtractConstant);
            if (ov::op::util::constantIsEqualTo(resultSubtractConstant, 0.f)) {
                resultSubtractConstant = nullptr;
            }
        }

        if (resultSubtractConstant == nullptr) {
            const auto multiply = opsMap.at(multiplyWrapper).get_node_shared_ptr();
            const auto newMultiply = std::make_shared<opset1::Multiply>(weightsConvert, opsMap.at(multiplyConstantWrapper).get_node_shared_ptr());
            NetworkHelper::copyInfo(multiply, newMultiply);
            replace_node(multiply, newMultiply);
        } else {
            NetworkHelper::copyInfo(subtractConstant, resultSubtractConstant);
            const auto resultConvert = std::make_shared<opset1::Convert>(resultSubtractConstant, dequantizationPrecision);
            NetworkHelper::copyInfo(subtractConstant, resultConvert);
            resultConvert->set_friendly_name(subtractConstant->get_friendly_name() + "/Convert");

            ov::disable_constant_folding(resultConvert);

            const auto newSubtract = std::make_shared<opset1::Subtract>(opsMap.at(weightsConvertWrapper).get_node_shared_ptr(), resultConvert);
            NetworkHelper::copyInfo(subtract, newSubtract);
            replace_node(subtract, newSubtract);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(multiplyWrapper, matcher_name);
    this->register_matcher(m, callback);
}
