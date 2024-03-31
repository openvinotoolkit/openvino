// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/model.hpp"

namespace ov {
namespace intel_cpu {

bool has_matmul_with_compressed_weights(const std::shared_ptr<const ov::Model>& model);

}   // namespace intel_cpu
}   // namespace ov
