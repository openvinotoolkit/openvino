// Copyright (C) 2019-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>
#include <memory>
#include <vector>

#include "openvino/util/idevice_monitor.hpp"
namespace ov::util {
std::map<std::string, float> get_device_utilization(const std::string& device_id);
}  // namespace ov::util