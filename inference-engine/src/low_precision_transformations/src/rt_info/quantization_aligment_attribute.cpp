// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/quantization_aligment_attribute.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "low_precision/network_helper.hpp"

using namespace ngraph;

template class ngraph::VariantImpl<QuantizationAligmentAttribute>;

constexpr VariantTypeInfo VariantWrapper<QuantizationAligmentAttribute>::type_info;

std::shared_ptr<ngraph::Variant> VariantWrapper<QuantizationAligmentAttribute>::merge(const ngraph::NodeVector& nodes) {
    std::shared_ptr<QuantizationAligmentAttribute::SharedPart> resultSharedPart;
    std::shared_ptr<QuantizationAligmentAttribute::SharedPart::SharedValue> resultValue;

    // update
    for (const std::shared_ptr<ngraph::Node>& node : nodes) {
        auto& rt = node->get_rt_info();
        auto rtIt = rt.find(VariantWrapper<QuantizationAligmentAttribute>::type_info.name);
        if (rtIt == rt.end()) {
            continue;
        }

        auto& attribute = std::dynamic_pointer_cast<VariantWrapper<QuantizationAligmentAttribute>>(rtIt->second);
        QuantizationAligmentAttribute expectedOperation = attribute->get();
        std::shared_ptr<QuantizationAligmentAttribute::SharedPart>& sharedPart = expectedOperation.sharedPart;

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
        auto rtIt = rt.find(VariantWrapper<QuantizationAligmentAttribute>::type_info.name);
        if (rtIt == rt.end()) {
            continue;
        }

        auto& attributeWrapper = std::dynamic_pointer_cast<VariantWrapper<QuantizationAligmentAttribute>>(rtIt->second);
        QuantizationAligmentAttribute attribute = attributeWrapper->get();
        attribute.sharedPart->value = resultValue;
    }

    auto newAttribute = std::make_shared<::ngraph::VariantWrapper<QuantizationAligmentAttribute>>(resultSharedPart);
    return newAttribute;
}

std::shared_ptr<ngraph::Variant> VariantWrapper<QuantizationAligmentAttribute>::init(const std::shared_ptr<ngraph::Node>& node) {
    return nullptr;
}

std::string VariantWrapper<QuantizationAligmentAttribute>::get_string() {
    auto value = this->m_value;
    return
        std::string("intervalLow: ") + std::to_string(value.sharedPart->value->intervalLow) +
        std::string(", intervalHigh: ") + std::to_string(value.sharedPart->value->intervalHigh);
}
