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

template class ngraph::VariantImpl<QuantizationAlignmentAttributePtr>;

constexpr VariantTypeInfo VariantWrapper<QuantizationAlignmentAttributePtr>::type_info;

std::shared_ptr<ngraph::Variant> VariantWrapper<QuantizationAlignmentAttributePtr>::merge(const ngraph::NodeVector& nodes) {
    std::shared_ptr<::ngraph::VariantWrapper<QuantizationAlignmentAttributePtr>> resultAttributeWrapper;
    std::shared_ptr<QuantizationAlignmentAttribute> resultAttribute;

    // update
    for (const std::shared_ptr<ngraph::Node>& node : nodes) {
        auto& rt = node->get_rt_info();
        auto rtIt = rt.find(VariantWrapper<QuantizationAlignmentAttributePtr>::type_info.name);
        if (rtIt == rt.end()) {
            continue;
        }

        auto attributeWrapper = std::dynamic_pointer_cast<VariantWrapper<QuantizationAlignmentAttributePtr>>(rtIt->second);
        auto attribute = attributeWrapper->get();

        if (resultAttributeWrapper == nullptr) {
            resultAttributeWrapper = attributeWrapper;
            resultAttribute = attribute;
            continue;
        }

        resultAttribute->hasToBeAligned = resultAttribute->hasToBeAligned || attribute->hasToBeAligned;
    }

    return resultAttributeWrapper;
}

std::shared_ptr<ngraph::Variant> VariantWrapper<QuantizationAlignmentAttributePtr>::init(const std::shared_ptr<ngraph::Node>& node) {
    return nullptr;
}

std::string VariantWrapper<QuantizationAlignmentAttributePtr>::get_string() {
    std::stringstream ss;

#ifdef _DEBUG
    const size_t rawPointer = (size_t)m_value.get();
    ss << rawPointer << ": ";
#endif
    ss << "value: " << (m_value->hasToBeAligned ? "true" : "false");
    return ss.str();
}
