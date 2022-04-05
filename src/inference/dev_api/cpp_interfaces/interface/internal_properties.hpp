// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced intranal properties for OpenVINO runtime devices
 *        To use in set_property, compile_model, import_model, get_property methods
 *
 * @file cpp_interfaces/interface/internal_properties.hpp
 */
#pragma once

#include "openvino/runtime/properties.hpp"

namespace ov {

/**
 * @brief Legacy sub set of properties used just by InferenceEngine API
 */
static constexpr NamedProperties legacy_property{"OPENVINO_LEGACY_PROPERTY"};

/**
 * @brief Sub set of properties that used both in OpenVINO and InferenceEngine APIs
 */
static constexpr NamedProperties common_property{"OPENVINO_COMMON_PROPERTY"};

/**
 * @brief Sub set of properties hidden from public API
 */
static constexpr NamedProperties internal_property{"OPENVINO_INTERNAL_PROPERTY"};

}  // namespace ov