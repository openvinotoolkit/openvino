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

}  // namespace intel_auto
}  // namespace ov