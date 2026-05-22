// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include "openvino/core/any.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/runtime_attribute.hpp"

namespace ov::intel_cpu {

constexpr const char* FakeQuantizeClampBoundsAttr = "FakeQuantizeClampBounds";

inline std::pair<float, float> compose_clamp_intervals(const float inner_low,
                                                       const float inner_high,
                                                       const float outer_low,
                                                       const float outer_high) {
    if (inner_high < outer_low) {
        return {outer_low, outer_low};
    }

    if (inner_low > outer_high) {
        return {outer_high, outer_high};
    }

    return {std::max(inner_low, outer_low), std::min(inner_high, outer_high)};
}

class FakeQuantizeClampBounds : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI(FakeQuantizeClampBoundsAttr);

    FakeQuantizeClampBounds() = default;
    FakeQuantizeClampBounds(const float low, const float high) : m_low(low), m_high(high) {}

    [[nodiscard]] float low() const {
        return m_low;
    }

    [[nodiscard]] float high() const {
        return m_high;
    }

    [[nodiscard]] std::string to_string() const override {
        std::ostringstream builder;
        builder << m_low << ':' << m_high;
        return builder.str();
    }

    [[nodiscard]] ov::Any merge(const ov::NodeVector& nodes) const override {
        std::optional<std::pair<float, float>> merged;

        for (const auto& node : nodes) {
            const auto& rt_info = node->get_rt_info();
            const auto it = rt_info.find(get_type_info_static());
            if (it == rt_info.end()) {
                continue;
            }

            const auto bounds = it->second.as<FakeQuantizeClampBounds>();
            if (!merged.has_value()) {
                merged = {bounds.low(), bounds.high()};
                continue;
            }

            OPENVINO_ASSERT(merged->first == bounds.low() && merged->second == bounds.high(),
                            "Conflicting FakeQuantize clamp bounds while merging runtime info");
        }

        OPENVINO_ASSERT(merged.has_value(), "No FakeQuantize clamp bounds to merge");
        return FakeQuantizeClampBounds(merged->first, merged->second);
    }

private:
    float m_low = 0.f;
    float m_high = 0.f;
};

inline std::optional<FakeQuantizeClampBounds> get_fake_quantize_clamp_bounds(
    const std::shared_ptr<const ov::Node>& node) {
    const auto& rt_info = node->get_rt_info();
    const auto it = rt_info.find(FakeQuantizeClampBounds::get_type_info_static());
    if (it == rt_info.end()) {
        return std::nullopt;
    }

    return it->second.as<FakeQuantizeClampBounds>();
}

inline void set_fake_quantize_clamp_bounds(const std::shared_ptr<ov::Node>& node,
                                           const float low,
                                           const float high) {
    node->get_rt_info()[FakeQuantizeClampBounds::get_type_info_static()] = FakeQuantizeClampBounds(low, high);
}

}  // namespace ov::intel_cpu