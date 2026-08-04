// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace tp_gpu {

/// \brief Number of tensor-parallel ranks.
///
/// Must match the size of `device_ids` when both are set. At least one of the
/// two has to be provided.
static constexpr Property<uint32_t> tp_size{"TP_SIZE"};

/// \brief Explicit per-rank device names, e.g. {"GPU.0", "GPU.1"}.
///
/// When omitted, ranks are mapped to GPU.0 .. GPU.{tp_size - 1}.
static constexpr Property<std::vector<std::string>> device_ids{"DEVICE_IDS"};

}  // namespace tp_gpu
}  // namespace ov
