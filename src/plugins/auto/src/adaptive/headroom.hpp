// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>
#include <string>
#include <unordered_map>

#include "observation.hpp"
#include "openvino/runtime/device_id_parser.hpp"

namespace ov {
namespace auto_plugin {

// The threshold of a device: the one set for its exact name (GPU.0), else the one set for its base name (GPU).
inline std::optional<unsigned> find_utilization_threshold(const std::unordered_map<std::string, unsigned>& thresholds,
                                                          const std::string& device_name) {
    if (thresholds.empty()) {
        return std::nullopt;
    }
    if (const auto exact = thresholds.find(device_name); exact != thresholds.end()) {
        return exact->second;
    }
    if (const auto base = thresholds.find(ov::DeviceIDParser(device_name).get_device_name());
        base != thresholds.end()) {
        return base->second;
    }
    return std::nullopt;
}

// How far the closest utilization is from its threshold in percentage points; negative once it is over.
inline std::optional<float> threshold_headroom(const Utilizations& utilizations,
                                               const std::unordered_map<std::string, unsigned>& thresholds) {
    std::optional<float> headroom;
    for (const auto& [device_name, utilization] : utilizations) {
        if (const auto threshold = find_utilization_threshold(thresholds, device_name)) {
            const auto distance = static_cast<float>(*threshold) - utilization;
            headroom = headroom ? std::min(*headroom, distance) : distance;
        }
    }
    return headroom;
}

}  // namespace auto_plugin
}  // namespace ov
