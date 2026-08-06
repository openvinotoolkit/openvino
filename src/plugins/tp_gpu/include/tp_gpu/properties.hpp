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

/// \brief Upper bound, in milliseconds, on how long a rank waits inside a
/// collective before the whole group is aborted.
///
/// A collective needs every rank to show up. If one rank dies, never reaches
/// the call, or its device work never completes, the remaining ranks would
/// otherwise block forever and take the calling process with them. When the
/// bound is exceeded the group is marked failed, every waiter is woken and
/// each of them throws; the coordinator stays failed afterwards because the
/// device queues are left in an unknown state.
///
/// Default: 5000. Set to 0 to wait indefinitely (debugging only).
static constexpr Property<uint32_t> communication_timeout_ms{"COMMUNICATION_TIMEOUT_MS"};

}  // namespace tp_gpu
}  // namespace ov
