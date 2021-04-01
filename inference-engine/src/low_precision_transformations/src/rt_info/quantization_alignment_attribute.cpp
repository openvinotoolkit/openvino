// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/quantization_alignment_attribute.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "low_precision/network_helper.hpp"

using namespace ngraph;

template class ngraph::VariantImpl<QuantizationAlignmentAttribute>;

constexpr VariantTypeInfo VariantWrapper<QuantizationAlignmentAttribute>::type_info;

std::shared_ptr<ngraph::Variant> VariantWrapper<QuantizationAlignmentAttribute>::merge(const ngraph::NodeVector& nodes) {
    std::shared_ptr<QuantizationAlignmentAttribute::SharedPart> resultSharedPart;
    std::shared_ptr<QuantizationAlignmentAttribute::SharedPart::SharedValue> resultValue;

    // update
    for (const std::shared_ptr<ngraph::Node>& node : nodes) {
        auto& rt = node->get_rt_info();
        auto rtIt = rt.find(VariantWrapper<QuantizationAlignmentAttribute>::type_info.name);
        if (rtIt == rt.end()) {
            continue;
        }

        auto& attribute = std::dynamic_pointer_cast<VariantWrapper<QuantizationAlignmentAttribute>>(rtIt->second);
        QuantizationAlignmentAttribute expectedOperation = attribute->get();
        std::shared_ptr<QuantizationAlignmentAttribute::SharedPart>& sharedPart = expectedOperation.sharedPart;

        if (resultValue == nullptr) {
            resultSharedPart = sharedPart;
            resultValue = sharedPart->value;
        } else {
            if (resultValue->intervalLow > sharedPart->value->intervalLow) {
                resultValue->intervalLow = sharedPart->value->intervalLow;
            }

            if (resultValue->intervalHigh < sharedPart->value->intervalHigh) {
                resultValue->intervalHigh = sharedPart->value->intervalHigh;
            }
        }
    }

    // assign
    for (const std::shared_ptr<ngraph::Node>& node : nodes) {
        auto& rt = node->get_rt_info();
        auto rtIt = rt.find(VariantWrapper<QuantizationAlignmentAttribute>::type_info.name);
        if (rtIt == rt.end()) {
            continue;
        }

        auto& attributeWrapper = std::dynamic_pointer_cast<VariantWrapper<QuantizationAlignmentAttribute>>(rtIt->second);
        QuantizationAlignmentAttribute attribute = attributeWrapper->get();
        attribute.sharedPart->value = resultValue;
    }

    auto newAttribute = std::make_shared<::ngraph::VariantWrapper<QuantizationAlignmentAttribute>>(resultSharedPart);
    return newAttribute;
}

std::shared_ptr<ngraph::Variant> VariantWrapper<QuantizationAlignmentAttribute>::init(const std::shared_ptr<ngraph::Node>& node) {
    return nullptr;
}

std::string VariantWrapper<QuantizationAlignmentAttribute>::get_string() {
    auto value = this->m_value.sharedPart->value;
    return
        std::string("low: ") + std::to_string(value->intervalLow) +
        std::string(", high: ") + std::to_string(value->intervalHigh) +
        std::string(", hasToBeAligned: ") + (value->hasToBeAligned ? "true" : "false");
}
