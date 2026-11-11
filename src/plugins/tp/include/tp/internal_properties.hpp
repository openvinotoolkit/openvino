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
#include "openvino/runtime/tp/collective_contract.hpp"
#include "tp/properties.hpp"

namespace ov {
namespace tp {

namespace internal {
/// Number of ranks of a TP compiled model.
static constexpr Property<int, PropertyMutability::RO> world_size{"TP_WORLD_SIZE"};
}  // namespace internal

// -----------------------------------------------------------------------------
// Tuning knobs. Available in every build type but not settable through the
// public API: they trade one performance profile for another. Reachable through
// the environment only (OV_ + the key below).
//
// The collectives are tuned in the GPU plugin, which runs them: GPU_TP_* options,
// e.g. OV_GPU_TP_ENABLE_HALVING.
// -----------------------------------------------------------------------------

/// Size ceiling in bytes for routing a user input through plugin-owned host
/// memory instead of a device staging copy. 0 always stages on the device.
static constexpr Property<uint64_t> input_stage_max_bytes{"TP_INPUT_STAGE_MAX_BYTES"};

/// Leave the vocabulary projection whole, so every rank computes the full lm_head and no gather is
/// inserted. Numerically the same; when the LM head shares its weight with the embedding, it saves
/// the cut copy (table / N per rank) at the cost of a slower decode step. Part of the cache key.
static constexpr Property<bool> disable_lm_head_sharding{"TP_DISABLE_LM_HEAD_SHARDING"};

// -----------------------------------------------------------------------------
// Debug options. Compiled out entirely unless ENABLE_TP_DEBUG_CAPS is on:
// without it the getters do not exist, the options are not registered at all,
// passing one throws, and the environment variable is ignored.
//
// They cover this plugin only. The device plugin that runs the collectives has
// options of its own, e.g. GPU_TP_VERBOSE / OV_GPU_TP_VERBOSE.
// -----------------------------------------------------------------------------

/// Verbosity of the plugin's diagnostics.
static constexpr Property<ov::log::Level> verbose{"TP_VERBOSE"};

/// File the plugin's diagnostics and reports go to instead of the standard output.
static constexpr Property<std::string> log_to_file{"TP_LOG_TO_FILE"};

/// Directory where compile_model writes the sharding verifier's state of every split or annotated node.
static constexpr Property<std::string> dump_sharding_states{"TP_DUMP_SHARDING_STATES"};

// -----------------------------------------------------------------------------
// Properties below produce silently wrong results. They exist to bisect
// where a discrepancy comes from, never to run a model.
// -----------------------------------------------------------------------------

/// Compile a single rank's shard with every collective stripped out. A shard
/// without its AllReduce is not the model: the output is wrong by
/// construction. Also changes the rank count and collective count recorded in
/// an exported blob.
static constexpr Property<bool> shard_only{"TP_SHARD_ONLY"};

}  // namespace tp
}  // namespace ov
