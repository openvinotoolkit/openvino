// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/model.hpp"

namespace intel_npu {

std::shared_ptr<ov::Model> runOVPasses(const std::shared_ptr<const ov::Model>& model);

}  // namespace intel_npu
