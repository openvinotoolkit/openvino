// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "intel_npu/common/network_metadata.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"

namespace intel_npu {

bool isInitMetadata(const NetworkMetadata& networkMetadata);

}  // namespace intel_npu
