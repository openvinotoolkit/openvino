// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/quantization_alignment_intervals_attribute.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "low_precision/network_helper.hpp"

using namespace ngraph;

template class ngraph::VariantImpl<QuantizationAlignmentIntervalsAttributePtr>;

constexpr VariantTypeInfo VariantWrapper<QuantizationAlignmentIntervalsAttributePtr>::type_info;

std::shared_ptr<ngraph::Variant> VariantWrapper<QuantizationAlignmentIntervalsAttributePtr>::merge(const ngraph::NodeVector& nodes) {
    std::shared_ptr<::ngraph::VariantWrapper<QuantizationAlignmentIntervalsAttributePtr>> resultAttributeWrapper;
    std::shared_ptr<QuantizationAlignmentIntervalsAttribute> resultAttribute;

    // update
    for (const std::shared_ptr<ngraph::Node>& node : nodes) {
        auto& rt = node->get_rt_info();
        auto rtIt = rt.find(VariantWrapper<QuantizationAlignmentIntervalsAttributePtr>::type_info.name);
        if (rtIt == rt.end()) {
            continue;
        }

        auto attributeWrapper = std::dynamic_pointer_cast<VariantWrapper<QuantizationAlignmentIntervalsAttributePtr>>(rtIt->second);
        auto attribute = attributeWrapper->get();

        if (resultAttributeWrapper == nullptr) {
            resultAttributeWrapper = attributeWrapper;
            resultAttribute = attribute;
            continue;
        }


        if (resultAttribute->intervalLow > attribute->intervalLow) {
            resultAttribute->intervalLow = attribute->intervalLow;
        }

        if (resultAttribute->intervalHigh < attribute->intervalHigh) {
            resultAttribute->intervalHigh = attribute->intervalHigh;
        }
    }

    return resultAttributeWrapper;
}

std::shared_ptr<ngraph::Variant> VariantWrapper<QuantizationAlignmentIntervalsAttributePtr>::init(const std::shared_ptr<ngraph::Node>& node) {
    return nullptr;
}

std::string VariantWrapper<QuantizationAlignmentIntervalsAttributePtr>::get_string() {
    std::stringstream ss;

    // TODO: debug only
    const size_t rawPointer = (size_t)m_value.get();
    ss << rawPointer << ": ";

    ss << "low: " << m_value->intervalLow << ", high: " << m_value->intervalHigh;
    return ss.str();
}
