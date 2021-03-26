// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/expected_operation_attribute.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "low_precision/network_helper.hpp"

using namespace ngraph;

template class ngraph::VariantImpl<ExpectedOperationAttribute>;

constexpr VariantTypeInfo VariantWrapper<ExpectedOperationAttribute>::type_info;

std::shared_ptr<ngraph::Variant> VariantWrapper<ExpectedOperationAttribute>::merge(const ngraph::NodeVector& nodes) {
    std::unordered_map<std::string, std::vector<std::shared_ptr<ExpectedOperationAttribute::SharedValue>>> sharedValuesByOperation;

    for (const std::shared_ptr<ngraph::Node>& node : nodes) {
        auto& rt = node->get_rt_info();
        auto rtIt = rt.find(VariantWrapper<ExpectedOperationAttribute>::type_info.name);
        if (rtIt == rt.end()) {
            continue;
        }

        auto& attribute = std::dynamic_pointer_cast<VariantWrapper<ExpectedOperationAttribute>>(rtIt->second);
        ExpectedOperationAttribute expectedOperation = attribute->get();
        std::shared_ptr<ExpectedOperationAttribute::SharedValue>& sharedValue = expectedOperation.sharedValue;

        auto collectedIt = sharedValuesByOperation.find(sharedValue->operationName);
        if (collectedIt == sharedValuesByOperation.end()) {
            sharedValuesByOperation.emplace(
                sharedValue->operationName,
                std::vector<std::shared_ptr<ExpectedOperationAttribute::SharedValue>>({ sharedValue }));
        } else {
            collectedIt->second.push_back(sharedValue);
        }
    }

    NGRAPH_CHECK(sharedValuesByOperation.size() <= 1ul, "Multi-operation is not supported");

    auto newAttribute = std::make_shared<::ngraph::VariantWrapper<ExpectedOperationAttribute>>(this->m_value.sharedValue);

    if (sharedValuesByOperation.empty()) {
        return nullptr;
    }

    auto it = sharedValuesByOperation.begin();
    auto& sharedValues = it->second;
    const bool value = std::any_of(
        sharedValues.begin(),
        sharedValues.end(),
        [](const std::shared_ptr<ExpectedOperationAttribute::SharedValue>& sharedValue) { return sharedValue->value; });
    sharedValues[0]->value = value;

    // TODO: not completed
    //for (size_t index = 1ul; index < sharedValues.size(); ++index) {
    //    sharedValues[index] = sharedValues[0];
    //}

    return newAttribute;
}

std::shared_ptr<ngraph::Variant> VariantWrapper<ExpectedOperationAttribute>::init(const std::shared_ptr<ngraph::Node>& node) {
    return nullptr;
}

std::string VariantWrapper<ExpectedOperationAttribute>::get_string() {
    auto value = this->m_value;
    return
        std::string("value: ") + (value.sharedValue->value ? "true" : "false") +
        (value.sharedValue->operationName.empty() ? "" : ", operation: " + value.sharedValue->operationName);
}
