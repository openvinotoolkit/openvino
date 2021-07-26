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
#include "low_precision/lpt_visibility.hpp"

namespace ngraph {
class IntervalsAlignmentAttribute;

class LP_TRANSFORMATIONS_API IntervalsAlignmentSharedValue : public SharedValue<IntervalsAlignmentAttribute> {
public:
    class Interval {
    public:
        Interval() = default;
        Interval(const float low, const float high) : low(low), high(high) {}
        float low;
        float high;
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
    size_t minLevels;
    // preferable precisions which are preferred by affected quantization operations to avoid zero points
    std::set<element::Type> preferablePrecisions;

#ifdef LPT_DEBUG
    std::string minLevelsOperation;
#endif
};

class LP_TRANSFORMATIONS_API IntervalsAlignmentAttribute : public SharedValueAttribute<IntervalsAlignmentSharedValue> {
public:
    IntervalsAlignmentAttribute() = default;
    IntervalsAlignmentAttribute(IntervalsAlignmentSharedValue::Interval combinedInterval, size_t levels);
    IntervalsAlignmentAttribute(
        const IntervalsAlignmentSharedValue::Interval combinedInterval,
        const size_t levels,
        const IntervalsAlignmentSharedValue::Interval minInterval,
        const size_t minLevels);

    // specify subgraph original levels
    size_t levels;
};

using IntervalsAlignmentAttributePtr = std::shared_ptr<IntervalsAlignmentAttribute>;

extern template class LP_TRANSFORMATIONS_API ngraph::VariantImpl<IntervalsAlignmentAttributePtr>;

template<>
class LP_TRANSFORMATIONS_API VariantWrapper<std::shared_ptr<IntervalsAlignmentAttribute>> :
    public VariantImpl<std::shared_ptr<IntervalsAlignmentAttribute>> {
public:
    static constexpr VariantTypeInfo type_info{ "LowPrecision::IntervalsAlignment", 0 };

    const VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<IntervalsAlignmentAttribute> get() const { return this->m_value; }

    static std::shared_ptr<VariantWrapper<std::shared_ptr<IntervalsAlignmentAttribute>>> create(
        const std::shared_ptr<ngraph::Node>& node,
        const AttributeParameters& params);
    void merge(std::vector<std::shared_ptr<VariantWrapper<std::shared_ptr<IntervalsAlignmentAttribute>>>>& attributes);
    std::string to_string() override;
};
} // namespace ngraph
