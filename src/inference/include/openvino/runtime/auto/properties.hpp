// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/runtime/properties.hpp>
#include <string>

namespace ov {

/**
 * @brief Namespace with Intel AUTO specific properties
 */
namespace intel_auto {
/**
 * @brief auto/multi device setting that enables performance improvement by binding buffer to hw infer request
 */
static constexpr Property<bool> device_bind_buffer{"DEVICE_BIND_BUFFER"};

/**
 * @brief auto device setting that enable/disable CPU as acceleration (or helper device) at the beginning
 */
static constexpr Property<bool> enable_startup_fallback{"ENABLE_STARTUP_FALLBACK"};

/**
 * @brief auto device setting that enable/disable runtime fallback to other devices when infer fails on current
 * selected device
 */
static constexpr Property<bool> enable_runtime_fallback{"ENABLE_RUNTIME_FALLBACK"};
}  // namespace intel_auto
}  // namespace ov