// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <string>

#include "openvino/core/except.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tp_gpu/properties.hpp"

namespace ov {
namespace tp_gpu {

enum class ProfilingMode : uint8_t {
    NONE = 0,    //!< No measurement at all.
    HOST = 1,    //!< Host-side timings only. Does not change the schedule.
    DEVICE = 2,  //!< Device-side kernel timestamps only. Changes the schedule.
    ALL = 3,     //!< Both.
};

inline std::ostream& operator<<(std::ostream& os, const ProfilingMode& mode) {
    switch (mode) {
    case ProfilingMode::NONE:
        return os << "NONE";
    case ProfilingMode::HOST:
        return os << "HOST";
    case ProfilingMode::DEVICE:
        return os << "DEVICE";
    case ProfilingMode::ALL:
        return os << "ALL";
    default:
        OPENVINO_THROW("[TP_GPU] Unsupported profiling mode");
    }
}

inline std::istream& operator>>(std::istream& is, ProfilingMode& mode) {
    std::string str;
    is >> str;
    if (str == "NONE") {
        mode = ProfilingMode::NONE;
    } else if (str == "HOST") {
        mode = ProfilingMode::HOST;
    } else if (str == "DEVICE") {
        mode = ProfilingMode::DEVICE;
    } else if (str == "ALL") {
        mode = ProfilingMode::ALL;
    } else {
        OPENVINO_THROW("[TP_GPU] Unsupported profiling mode: ", str);
    }
    return is;
}

// -----------------------------------------------------------------------------
// Tuning knobs. Available in every build type but not settable through the
// public API: they trade one performance profile for another. Reachable through
// the environment (OV_ + the key below) and through the plugin config file.
// -----------------------------------------------------------------------------

/// Recursive halving/doubling instead of the ring for small payloads on
/// power-of-two world sizes. Halving takes 2*log2(N) dependency rounds
/// against the ring's 2*(N-1), which is what decode-sized payloads care
/// about; above `halving_max_bytes` the trade flips back to the ring.
static constexpr Property<bool> enable_halving{"TP_ENABLE_HALVING"};

/// Payload ceiling in bytes above which halving falls back to the ring.
static constexpr Property<uint64_t> halving_max_bytes{"TP_HALVING_MAX_BYTES"};

/// Size ceiling in bytes for routing a user input through plugin-owned host
/// memory instead of a device staging copy. 0 always stages on the device.
static constexpr Property<uint64_t> input_stage_max_bytes{"TP_INPUT_STAGE_MAX_BYTES"};

// -----------------------------------------------------------------------------
// Debug options. Compiled out entirely unless ENABLE_TP_GPU_DEBUG_CAPS is on:
// without it the getters do not exist, the options are not registered at all,
// passing one throws, and the environment variable is ignored.
// -----------------------------------------------------------------------------

/// Verbosity of the plugin's diagnostics.
static constexpr Property<ov::log::Level> verbose{"TP_VERBOSE"};

/// Performance profiling configurable via ProfilingMode.
static constexpr Property<ProfilingMode> profiling{"TP_PROFILING"};

/// Rank-0 collectives between two measurement dumps.
static constexpr Property<uint64_t> dump_period{"TP_DUMP_PERIOD"};

/// Force collectives off the spliced model queue onto their own queue with a
/// full drain. Much slower, the same code path is taken automatically when
/// the driver lacks the command-list splice extension.
static constexpr Property<bool> force_sync_collective{"TP_FORCE_SYNC_COLLECTIVE"};

/// Leave the vocabulary projection unsharded, so every rank computes the full
/// lm_head and no gather is inserted. Numerically correct either way; it
/// trades compute for one fewer collective. Changes the collective count
/// baked into an exported blob.
static constexpr Property<bool> disable_lm_head_sharding{"TP_DISABLE_LM_HEAD_SHARDING"};

// -----------------------------------------------------------------------------
// Properties below produce silently wrong results. They exist to bisect
// where a discrepancy comes from, never to run a model.
// -----------------------------------------------------------------------------

/// Compile a single rank's shard with every collective stripped out. A shard
/// without its AllReduce is not the model: the output is wrong by
/// construction. Also changes the rank count and collective count recorded in
/// an exported blob.
static constexpr Property<bool> shard_only{"TP_SHARD_ONLY"};

/// Return from every AllReduce without doing anything. Output buffers keep
/// whatever they held.
static constexpr Property<bool> skip_collective{"TP_SKIP_COLLECTIVE"};

}  // namespace tp_gpu
}  // namespace ov
