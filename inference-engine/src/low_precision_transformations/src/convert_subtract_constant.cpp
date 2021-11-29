// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/convert_subtract_constant.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "low_precision/network_helper.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::ConvertSubtractConstant, "ConvertSubtractConstant", 0);

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
ngraph::pass::low_precision::ConvertSubtractConstant::ConvertSubtractConstant(const std::vector<ngraph::element::Type>& constantPrecisions) {
    auto weightsConstantWrapper = ngraph::pattern::wrap_type<opset1::Constant>(pattern::consumers_count(1));
    auto weightsConvertWrapper = ngraph::pattern::wrap_type<opset1::Convert>({ weightsConstantWrapper }, pattern::consumers_count(1));
    auto subtractConstantWrapper = ngraph::pattern::wrap_type<opset1::Constant>(pattern::consumers_count(1));
    auto subtractWrapper = ngraph::pattern::wrap_type<opset1::Subtract>({ weightsConvertWrapper, subtractConstantWrapper }, pattern::consumers_count(1));
    auto multiplyConstantWrapper = ngraph::pattern::wrap_type<opset1::Constant>(pattern::consumers_count(1));
    auto multiplyWrapper = ngraph::pattern::wrap_type<opset1::Multiply>({ subtractWrapper, multiplyConstantWrapper }, pattern::consumers_count(1));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher & m) -> bool {
        const auto& opsMap = m.get_pattern_value_map();
        const auto weightsConvert = opsMap.at(weightsConvertWrapper).get_node_shared_ptr();
        const auto quantizePrecision = weightsConvert->get_input_element_type(0);
        const auto dequantizationPrecision = weightsConvert->get_output_element_type(0);

        // validation by Convert operation input precisions
        if (!constantPrecisions.empty()) {
            const ngraph::element::Type inputPrecision = quantizePrecision;
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
            if (op::util::constantIsEqualTo(resultSubtractConstant, 0.f)) {
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

            auto& rtInfo = resultConvert->get_rt_info();
            rtInfo["DISABLED_CONSTANT_FOLDING"] = std::make_shared<VariantWrapper<std::string>>("");

            const auto newSubtract = std::make_shared<opset1::Subtract>(opsMap.at(weightsConvertWrapper).get_node_shared_ptr(), resultConvert);
            NetworkHelper::copyInfo(subtract, newSubtract);
            replace_node(subtract, newSubtract);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(multiplyWrapper, "ConvertSubtractConstant");
    this->register_matcher(m, callback);
}
