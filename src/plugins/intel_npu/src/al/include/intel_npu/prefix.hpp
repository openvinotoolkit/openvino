// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <string_view>

namespace intel_npu {

//
// TODO
//
#define READVALUE_PREFIX           std::string("vpux_ie_read_value_")
#define ASSIGN_PREFIX              std::string("vpux_ie_assign_")
#define SHAPE_TENSOR_PREFIX        std::string("vpux_ie_shape_")
#define INIT_INPUT_WEIGHTS_PREFIX  std::string("in_ngraphSharedConstant_")
#define INIT_OUTPUT_WEIGHTS_PREFIX std::string("out_ngraphSharedConstant_")
#define MAIN_INPUT_WEIGHTS_PREFIX  std::string("out_ngraphSharedConstant_")

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

inline std::string stateOutputToStateInputName(std::string_view name) {
    return READVALUE_PREFIX + std::string(name.substr(ASSIGN_PREFIX.length()));
}

}  // namespace intel_npu
