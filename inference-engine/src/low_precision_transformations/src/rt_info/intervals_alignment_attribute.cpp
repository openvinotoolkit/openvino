// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/intervals_alignment_attribute.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass::low_precision;

IntervalsAlignmentAttribute::IntervalsAlignmentAttribute(const float intervalLow, const float intervalHigh, const size_t levels) : levels(levels) {
    sharedValue = std::make_shared<IntervalsAlignmentSharedValue>(intervalLow, intervalHigh);
}

template class ngraph::VariantImpl<IntervalsAlignmentAttributePtr>;

constexpr VariantTypeInfo VariantWrapper<IntervalsAlignmentAttributePtr>::type_info;

std::shared_ptr<ngraph::Variant> VariantWrapper<IntervalsAlignmentAttributePtr>::merge(const ngraph::NodeVector& nodes) {
    std::shared_ptr<::ngraph::VariantWrapper<IntervalsAlignmentAttributePtr>> resultAttributeWrapper;
    std::shared_ptr<IntervalsAlignmentAttribute> resultAttribute;

    // update
    for (const std::shared_ptr<ngraph::Node>& node : nodes) {
        auto& rt = node->get_rt_info();
        auto rtIt = rt.find(VariantWrapper<IntervalsAlignmentAttributePtr>::type_info.name);
        if (rtIt == rt.end()) {
            continue;
        }

        auto attributeWrapper = std::dynamic_pointer_cast<VariantWrapper<IntervalsAlignmentAttributePtr>>(rtIt->second);
        auto attribute = attributeWrapper->get();

        if (resultAttributeWrapper == nullptr) {
            resultAttributeWrapper = attributeWrapper;
            resultAttribute = attribute;
            continue;
        }


        if (resultAttribute->sharedValue->intervalLow > attribute->sharedValue->intervalLow) {
            resultAttribute->sharedValue->intervalLow = attribute->sharedValue->intervalLow;
        }

        if (resultAttribute->sharedValue->intervalHigh < attribute->sharedValue->intervalHigh) {
            resultAttribute->sharedValue->intervalHigh = attribute->sharedValue->intervalHigh;
        }
    }

    return resultAttributeWrapper;
}

std::shared_ptr<VariantWrapper<std::shared_ptr<IntervalsAlignmentAttribute>>> VariantWrapper<IntervalsAlignmentAttributePtr>::create(
    const std::shared_ptr<ngraph::Node>& node,
    const AttributeParameters& params) {
    if (!is_type<opset1::FakeQuantize>(node)) {
        return nullptr;
    }

    auto fakeQuantize = as_type_ptr<opset1::FakeQuantize>(node);
    if (!QuantizationDetails::outputLayoutIsSupported(fakeQuantize) || !QuantizationDetails::isSupportedLevel(fakeQuantize->get_levels())) {
        return nullptr;
    }

    float lowInterval;
    float highInterval;
    FakeQuantizeDequantization dequantization;
    {
        const auto targetInputs = node->output(0).get_target_inputs();
        if (targetInputs.size() == 1ul) {
            dequantization = NetworkHelper::getDequantizationBelow(node);
        }
    }

    const auto outLow = as_type_ptr<opset1::Constant>(node->get_input_node_shared_ptr(3));
    const auto outHigh = as_type_ptr<opset1::Constant>(node->get_input_node_shared_ptr(4));
    if (!NetworkHelper::isScalarLike(outLow) || !NetworkHelper::isScalarLike(outHigh)) {
        return nullptr;
    }

    if (dequantization.empty()) {
        const std::vector<float> lowIntervals = outLow->cast_vector<float>();
        lowInterval = *std::min_element(lowIntervals.begin(), lowIntervals.end());

        const std::vector<float> highIntervals = outHigh->cast_vector<float>();
        highInterval = *std::max_element(highIntervals.begin(), highIntervals.end());
    } else {
        {
            auto multiplyResult = dequantization.multiplyConstant == nullptr ?
                                  node->get_input_node_ptr(3)->shared_from_this() :
                                  fold<opset1::Multiply>(
                                              foldConvert(node->get_input_node_ptr(3)->shared_from_this(), params.deqPrecision),
                                          dequantization.multiplyConstant);

            auto multiplyResultConstant = as_type_ptr<opset1::Constant>(multiplyResult);
            auto intervals = multiplyResultConstant->cast_vector<float>();
            lowInterval = *std::min_element(intervals.begin(), intervals.end());
        }

        {
            auto multiplyResult = dequantization.multiplyConstant == nullptr ?
                                  node->get_input_node_ptr(4)->shared_from_this() :
                                  fold<opset1::Multiply>(
                                          foldConvert(node->get_input_node_ptr(4)->shared_from_this(), params.deqPrecision),
                                          dequantization.multiplyConstant);

            auto multiplyResultConstant = as_type_ptr<opset1::Constant>(multiplyResult);
            auto intervals = multiplyResultConstant->cast_vector<float>();
            highInterval = *std::max_element(intervals.begin(), intervals.end());
        }
    }

    auto& rtInfo = node->get_rt_info();
    const auto attribute = std::make_shared<::ngraph::VariantWrapper<IntervalsAlignmentAttributePtr>>(
        ngraph::pass::low_precision::make_shared_attribute<IntervalsAlignmentAttribute>(lowInterval, highInterval, fakeQuantize->get_levels()));
    rtInfo[ngraph::VariantWrapper<IntervalsAlignmentAttributePtr>::type_info.name] = attribute;

    return attribute;
}

void VariantWrapper<IntervalsAlignmentAttributePtr>::merge(
    std::vector<std::shared_ptr<VariantWrapper<std::shared_ptr<IntervalsAlignmentAttribute>>>>& attributes) {
    std::shared_ptr<IntervalsAlignmentAttribute> resultAttribute = get();
    for (const auto& attributeWrapper : attributes) {
        auto attribute = attributeWrapper->get();

        if (resultAttribute->sharedValue->intervalLow > attribute->sharedValue->intervalLow) {
            resultAttribute->sharedValue->intervalLow = attribute->sharedValue->intervalLow;
        }

        if (resultAttribute->sharedValue->intervalHigh < attribute->sharedValue->intervalHigh) {
            resultAttribute->sharedValue->intervalHigh = attribute->sharedValue->intervalHigh;
        }
    }
}

std::string VariantWrapper<IntervalsAlignmentAttributePtr>::get_string() {
    std::stringstream ss;
    ss << m_value->get_string();
    ss << (m_value->levels == 0ul ? "" : ("levels: " + std::to_string(m_value->levels) + ", ")) <<
        "low: " << m_value->sharedValue->intervalLow <<
        ", high: " << m_value->sharedValue->intervalHigh;
    return ss.str();
}
