// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>

#include "openvino/runtime/properties.hpp"

#ifdef MULTIUNITTEST
#define auto_plugin mock_auto_plugin
#endif

namespace ov {
namespace auto_plugin {

// How far AUTO may decide on its own at run time: Off keeps the static behavior, Route may switch between the
// instances that are already compiled, Full may also compile new ones.
enum class AdaptiveLevel { Off, Route, Full };

// Private to the AUTO plugin: not part of the public API and not listed in the supported properties.
inline constexpr ov::Property<std::string> adaptive_level{"AUTO_ADAPTIVE_LEVEL"};

inline std::optional<AdaptiveLevel> parse_adaptive_level(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    if (value == "OFF")
        return AdaptiveLevel::Off;
    if (value == "ROUTE")
        return AdaptiveLevel::Route;
    if (value == "FULL")
        return AdaptiveLevel::Full;
    return std::nullopt;
}

inline const char* to_string(AdaptiveLevel level) {
    switch (level) {
    case AdaptiveLevel::Route:
        return "ROUTE";
    case AdaptiveLevel::Full:
        return "FULL";
    default:
        return "OFF";
    }
}

// What the decisions are made for, read by every policy.
struct AdaptiveGoal {
    AdaptiveLevel level = AdaptiveLevel::Off;
};

}  // namespace auto_plugin
}  // namespace ov
