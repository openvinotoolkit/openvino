// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <string_view>

namespace intel_npu {

//
// Prefixes used to identify special inputs/outputs
//

constexpr std::string_view READVALUE_PREFIX = "vpux_ie_read_value_";
constexpr std::string_view ASSIGN_PREFIX = "vpux_ie_assign_";
constexpr std::string_view SHAPE_TENSOR_PREFIX = "vpux_ie_shape_";
constexpr std::string_view INIT_INPUT_WEIGHTS_PREFIX = "vpux_ow_";
constexpr std::string_view INIT_OUTPUT_WEIGHTS_PREFIX = "vpux_tw_";
constexpr std::string_view MAIN_INPUT_WEIGHTS_PREFIX = "vpux_tw_";

inline bool nameHasPrefix(std::string_view name, std::string_view prefix) {
    return !name.compare(0, prefix.length(), prefix);
}

inline bool isStateInputName(std::string_view name) {
    return nameHasPrefix(name, READVALUE_PREFIX);
}
inline bool isStateOutputName(std::string_view name) {
    return nameHasPrefix(name, ASSIGN_PREFIX);
}
inline bool isShapeTensorName(std::string_view name) {
    return nameHasPrefix(name, SHAPE_TENSOR_PREFIX);
}

inline bool isInitInputWeightsName(std::string_view name) {
    return nameHasPrefix(name, INIT_INPUT_WEIGHTS_PREFIX);
}
inline bool isInitOutputWeightsName(std::string_view name) {
    return nameHasPrefix(name, INIT_OUTPUT_WEIGHTS_PREFIX);
}
inline bool isMainInputWeightsName(std::string_view name) {
    return nameHasPrefix(name, MAIN_INPUT_WEIGHTS_PREFIX);
}

}  // namespace intel_npu
