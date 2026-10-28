// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace tp {

/// Tuning and diagnostics of the collectives, resolved by whoever owns the configuration.
struct CollectiveOptions {
    bool enable_halving = true;
    uint64_t halving_max_bytes = 256 * 1024;
    bool force_sync_collective = false;
    bool skip_collective = false;
    bool profiling_host = false;
    bool profiling_device = false;
    std::size_t dump_period = 0;  ///< collectives between two reports; 0 when profiling is off
    ov::log::Level verbosity = ov::log::Level::NO;
};

}  // namespace tp
}  // namespace ov
