// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include "low_precision/rt_info/shared_value_attribute.hpp"
#include "low_precision/rt_info/attribute_parameters.hpp"
#include "openvino/core/ov_visibility.hpp"

namespace ngraph {
class OPENVINO_API IntervalsAlignmentSharedValue {
public:
    class Interval {
    public:
        Interval() = default;
        Interval(const float low, const float high) : low(low), high(high) {}
        float low = 0.f;
        float high = 0.f;
    };

    IntervalsAlignmentSharedValue() = default;
    IntervalsAlignmentSharedValue(
        const Interval& combinedInterval,
        const Interval& minInterval,
        const size_t minLevels) :
        combinedInterval(combinedInterval),
        minInterval(minInterval),
        minLevels(minLevels) {}

    Interval combinedInterval;
    Interval minInterval;
    size_t minLevels = 0;
    // preferable precisions which are preferred by affected quantization operations to avoid zero points
    std::set<element::Type> preferablePrecisions;

#ifdef LPT_DEBUG
    std::string minLevelsOperation;
#endif
};

class OPENVINO_API IntervalsAlignmentAttribute : public SharedAttribute<IntervalsAlignmentSharedValue> {
public:
    OPENVINO_RTTI("LowPrecision::IntervalsAlignment", "", ov::RuntimeAttribute, 0);
    IntervalsAlignmentAttribute() = default;
    IntervalsAlignmentAttribute(IntervalsAlignmentSharedValue::Interval combinedInterval, size_t levels);
    IntervalsAlignmentAttribute(
        const IntervalsAlignmentSharedValue::Interval combinedInterval,
        const size_t levels,
        const IntervalsAlignmentSharedValue::Interval minInterval,
        const size_t minLevels);

    static ov::Any create(
        const std::shared_ptr<ngraph::Node>& node,
        const AttributeParameters& params);
    void merge(std::vector<ov::Any>& attributes);
    std::string to_string() const override;

    // specify subgraph original levels
    size_t levels;
};

} // namespace ngraph
