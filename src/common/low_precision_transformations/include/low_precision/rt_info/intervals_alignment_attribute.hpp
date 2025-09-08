// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/core/node.hpp"

#include "low_precision/rt_info/shared_value_attribute.hpp"
#include "low_precision/rt_info/attribute_parameters.hpp"
#include "low_precision/lpt_visibility.hpp"

namespace ov {
/**
 * @ingroup ov_transformation_common_api
 * @brief IntervalsAlignmentSharedValue is used by IntervalsAlignmentAttribute as attribute shared value.
 */
class LP_TRANSFORMATIONS_API IntervalsAlignmentSharedValue {
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

/**
 * @ingroup ov_transformation_common_api
 * @brief IntervalsAlignmentAttribute defines subgraph with the same quantization intervals alignment.
 * FakeQuantize operations are included. The attribute is used by quantization operations.
 *
 * For more details about the attribute, refer to
 * [IntervalsAlignmentAttribute](@ref openvino_docs_OV_UG_lpt_IntervalsAlignment) page in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API IntervalsAlignmentAttribute : public SharedAttribute<IntervalsAlignmentSharedValue> {
public:
    OPENVINO_RTTI("LowPrecision::IntervalsAlignment", "", ov::RuntimeAttribute);
    IntervalsAlignmentAttribute() = default;
    IntervalsAlignmentAttribute(IntervalsAlignmentSharedValue::Interval combinedInterval, size_t levels);
    IntervalsAlignmentAttribute(
        const IntervalsAlignmentSharedValue::Interval combinedInterval,
        const size_t levels,
        const IntervalsAlignmentSharedValue::Interval minInterval,
        const size_t minLevels);

    static ov::Any create(
        const std::shared_ptr<ov::Node>& node,
        const AttributeParameters& params = AttributeParameters());
    void merge_attributes(std::vector<ov::Any>& attributes);
    std::string to_string() const override;

    // specify subgraph original levels
    size_t levels;
};

} // namespace ov
