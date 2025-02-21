// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/intervals_alignment_attribute.hpp"

#include <memory>
#include <string>
#include <vector>

#include "low_precision/lpt_itt.hpp"
#include "low_precision/network_helper.hpp"

using namespace ov;
using namespace ov;
using namespace ov::pass::low_precision;

IntervalsAlignmentAttribute::IntervalsAlignmentAttribute(
    const IntervalsAlignmentSharedValue::Interval combinedInterval,
    size_t levels) :
    SharedAttribute(IntervalsAlignmentSharedValue{combinedInterval, combinedInterval, levels}),
    levels(levels) {
}

IntervalsAlignmentAttribute::IntervalsAlignmentAttribute(
    const IntervalsAlignmentSharedValue::Interval combinedInterval,
    const size_t levels,
    const IntervalsAlignmentSharedValue::Interval minInterval,
    const size_t minLevels) :
    SharedAttribute(IntervalsAlignmentSharedValue{combinedInterval, minInterval, minLevels}),
    levels(levels) {
}

ov::Any IntervalsAlignmentAttribute::create(
    const std::shared_ptr<ov::Node>& node,
    const AttributeParameters& params) {
    if (!ov::is_type<opset1::FakeQuantize>(node)) {
        return nullptr;
    }

    auto fakeQuantize = ov::as_type_ptr<opset1::FakeQuantize>(node);
    if (!QuantizationDetails::outputLayoutIsSupported(fakeQuantize) || !QuantizationDetails::isSupportedLevel(fakeQuantize->get_levels())) {
        return nullptr;
    }

    float lowInterval;
    float highInterval;
    {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "calculateIntervals");

        FakeQuantizeDequantization dequantization;
        {
            const auto targetInputs = node->get_output_target_inputs(0);
            if (targetInputs.size() == 1ul) {
                dequantization = NetworkHelper::getDequantizationBelow(node, true);
            }
        }

        const auto outLow = ov::as_type_ptr<opset1::Constant>(node->get_input_node_shared_ptr(3));
        const auto outHigh = ov::as_type_ptr<opset1::Constant>(node->get_input_node_shared_ptr(4));
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
                        foldConvert(node->input_value(3), params.deqPrecision),
                        dequantization.multiplyConstant);

                auto multiplyResultConstant = ov::as_type_ptr<opset1::Constant>(multiplyResult);
                auto intervals = multiplyResultConstant->cast_vector<float>();
                lowInterval = *std::min_element(intervals.begin(), intervals.end());
            }

            {
                auto multiplyResult = dequantization.multiplyConstant == nullptr ?
                    node->get_input_node_ptr(4)->shared_from_this() :
                    fold<opset1::Multiply>(
                        foldConvert(node->input_value(4), params.deqPrecision),
                        dequantization.multiplyConstant);

                auto multiplyResultConstant = ov::as_type_ptr<opset1::Constant>(multiplyResult);
                auto intervals = multiplyResultConstant->cast_vector<float>();
                highInterval = *std::max_element(intervals.begin(), intervals.end());
            }
        }

        if (std::isinf(lowInterval) || std::isinf(highInterval)) {
            return nullptr;
        }
    }

    {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "create");

        assert(!std::isinf(lowInterval));
        assert(!std::isinf(highInterval));

        auto& rtInfo = node->get_rt_info();
        const IntervalsAlignmentSharedValue::Interval interval{ lowInterval, highInterval };
        auto attribute = IntervalsAlignmentAttribute(
                interval,
                fakeQuantize->get_levels());
        auto& result = (rtInfo[IntervalsAlignmentAttribute::get_type_info_static()] = attribute);

        const std::vector<float> outputLowValues = ov::as_type_ptr<opset1::Constant>(fakeQuantize->get_input_node_shared_ptr(3))->cast_vector<float>();
        const std::vector<float> outputHighValues = ov::as_type_ptr<opset1::Constant>(fakeQuantize->get_input_node_shared_ptr(4))->cast_vector<float>();
        LayerTransformation::PrecisionDetails preferablePrecision = LayerTransformation::getPrecisionDetails(
            fakeQuantize->get_levels(),
            outputLowValues,
            outputHighValues);
        if (preferablePrecision.precision != element::dynamic) {
            attribute.value().preferablePrecisions.insert(preferablePrecision.precision);
        }

#ifdef LPT_DEBUG
        attribute.value().minLevelsOperation = node->get_friendly_name();
#endif

        return result;
    }
}

void IntervalsAlignmentAttribute::merge_attributes(
    std::vector<ov::Any>& attributes) {
    for (const auto& attributeWrapper : attributes) {
        auto attribute = attributeWrapper.as<IntervalsAlignmentAttribute>().attribute;

        // TODO: LPT: copy/past: merge()
        auto& resultSharedValue = value();
        auto& sharedValue = attributeWrapper.as<IntervalsAlignmentAttribute>().value();
        if (levels != attributeWrapper.as<IntervalsAlignmentAttribute>().levels) {
            // TODO: LPT: not supported right now
            levels = 0ul;
            resultSharedValue.minLevels = 0ul;
        }

        if (resultSharedValue.combinedInterval.low > sharedValue.combinedInterval.low) {
            resultSharedValue.combinedInterval.low = sharedValue.combinedInterval.low;
        }

        if (resultSharedValue.combinedInterval.high < sharedValue.combinedInterval.high) {
            resultSharedValue.combinedInterval.high = sharedValue.combinedInterval.high;
        }

        assert(!std::isinf(resultSharedValue.combinedInterval.low));
        assert(!std::isinf(resultSharedValue.combinedInterval.high));

        resultSharedValue.preferablePrecisions.insert(sharedValue.preferablePrecisions.begin(), sharedValue.preferablePrecisions.end());

        const auto resultSize = std::abs(resultSharedValue.minInterval.high - resultSharedValue.minInterval.low);
        const auto size = std::abs(sharedValue.minInterval.high - sharedValue.minInterval.low);
        if (resultSize > size) {
            resultSharedValue.minInterval = sharedValue.minInterval;
            if (levels != 0ul) {
                float dequantizationMul;
                float dequantizationSub;
                float updatedOutputLowValue;
                float updatedOutputHighValue;

                const size_t minLevels = NetworkHelper::calculateLevels(
                        0.f,
                        DataPrecision::getMaxValue(levels),
                        resultSharedValue.combinedInterval.low,
                        resultSharedValue.combinedInterval.high,
                        resultSharedValue.minInterval.low,
                        resultSharedValue.minInterval.high,
                        dequantizationMul,
                        dequantizationSub,
                        updatedOutputLowValue,
                        updatedOutputHighValue);

                resultSharedValue.minLevels = minLevels;
            }

#ifdef LPT_DEBUG
            resultSharedValue.minLevelsOperation = sharedValue.minLevelsOperation;
#endif
        }
    }
}

std::string IntervalsAlignmentAttribute::to_string() const {
    std::stringstream preferablePrecisions;
    preferablePrecisions << "{";
    size_t index = 0;
    for (const auto& precision : value().preferablePrecisions) {
        preferablePrecisions << (index > 0 ? ", " : "") << precision;
        ++index;
    }
    preferablePrecisions << "}";

    std::stringstream ss;
    ss << attribute->get_string();
    ss << "levels: " + std::to_string(levels) << ", " <<
        "combined: { " << value().combinedInterval.low << ", " << value().combinedInterval.high << " }, " <<
        "min: { " << value().minInterval.low << ", " << value().minInterval.high << " }, "
        "minLevels: " << value().minLevels <<
#ifdef LPT_DEBUG
        ", minLevelsOperation: " << value().minLevelsOperation <<
#endif
        ", preferablePrecisions: " << preferablePrecisions.str();
    return ss.str();
}
