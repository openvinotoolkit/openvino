// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include "ie_system_conf.h"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include <exec_graph_info.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include "ie_system_conf.h"

namespace CPUTestUtils {
std::vector<CPUSpecificParams> filterCPUInfo(const std::vector<CPUSpecificParams>& CPUParams);
std::vector<CPUSpecificParams> filterCPUInfoForArch(const std::vector<CPUSpecificParams>& CPUParams);
std::vector<CPUSpecificParams> filterCPUInfoForDevice(const std::vector<CPUSpecificParams>& CPUParams);
} // namespace CPUTestUtils
