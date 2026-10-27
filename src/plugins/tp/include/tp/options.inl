// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// OV_CONFIG_RELEASE_OPTION:
//      Options exposed via the public API in all build types.
// OV_CONFIG_RELEASE_INTERNAL_OPTION:
//      Available in all build types, but not settable via the public API.
//      Reachable through the environment and the config file only.
// OV_CONFIG_TP_DEBUG_OPTION / OV_CONFIG_TP_DEBUG_GLOBAL_OPTION:
//      Only present when the plugin is built with ENABLE_TP_DEBUG_CAPS.
//      These are TP wrappers rather than the OV_CONFIG_DEBUG_* macros: the
//      stock ones key off ENABLE_DEBUG_CAPS, which would let the options exist
//      -- and accept an environment value -- in a build where the call sites
//      have been compiled out, so the value would be taken and then ignored.
//
// The environment variable name of an option is "OV_" + its property key, so
// `TP_HALVING_MAX_BYTES` is set through `OV_TP_HALVING_MAX_BYTES`.

// Namespace, property name, default value, [validator], description
OV_CONFIG_RELEASE_OPTION(ov::device, priorities, "", "Rank devices, one rank per entry in rank order; set through the device name, e.g. \"TP:GPU.0,GPU.1\"")
OV_CONFIG_RELEASE_OPTION(ov::tp, communication_timeout_ms, 5000u, "Milliseconds a rank waits inside a collective before the whole group is aborted. 0 waits indefinitely")

OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::tp, enable_halving, true, "Use recursive halving/doubling instead of the ring for small payloads on power-of-two world sizes")
OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::tp, halving_max_bytes, uint64_t{256 * 1024}, "Payload ceiling in bytes above which halving falls back to the ring")
OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::tp, input_stage_max_bytes, uint64_t{4096}, "Size ceiling in bytes for staging a user input through plugin-owned host memory. 0 always stages on the device")

OV_CONFIG_TP_DEBUG_GLOBAL_OPTION(ov::tp, verbose, ov::log::Level::NO, "Verbosity of the plugin's diagnostics. Values: LOG_NONE, LOG_ERROR, LOG_WARNING, LOG_INFO, LOG_DEBUG, LOG_TRACE")

OV_CONFIG_TP_DEBUG_OPTION(ov::tp, profiling, ov::tp::ProfilingMode::NONE, "Configurable performance profiling. Profiling modes: NONE, HOST, DEVICE, ALL.")
OV_CONFIG_TP_DEBUG_OPTION(ov::tp, dump_period, uint64_t{0}, "Rank-0 collectives between two measurement dumps. 0 means report every one")
OV_CONFIG_TP_DEBUG_OPTION(ov::tp, force_sync_collective, false, "Force collectives off the spliced model queue onto their own queue with a full drain")
OV_CONFIG_TP_DEBUG_OPTION(ov::tp, disable_lm_head_sharding, false, "Leave the vocabulary projection unsharded. Changes the collective count baked into an exported blob")
OV_CONFIG_TP_DEBUG_OPTION(ov::tp, shard_only, false, "DANGEROUS: compile one rank's shard with every collective stripped. Output is wrong by construction")
OV_CONFIG_TP_DEBUG_OPTION(ov::tp, skip_collective, false, "DANGEROUS: return from every AllReduce without doing anything. Output buffers keep whatever they held")
