// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/model.hpp"

namespace intel_npu {

void runOVPasses(const std::shared_ptr<ov::Model>& model);

}  // namespace intel_npu
