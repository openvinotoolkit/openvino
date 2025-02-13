// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/model.hpp"

namespace ov::intel_cpu {

bool has_matmul_with_compressed_weights(const std::shared_ptr<const ov::Model>& model);

}  // namespace ov::intel_cpu
