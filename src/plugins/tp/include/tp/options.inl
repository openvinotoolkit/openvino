// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// OV_CONFIG_RELEASE_OPTION:
//      Options exposed via the public API in all build types.
// OV_CONFIG_RELEASE_INTERNAL_OPTION:
//      Available in all build types, but not settable via the public API.
//      Reachable through the environment only.
// OV_CONFIG_TP_DEBUG_OPTION / OV_CONFIG_TP_DEBUG_GLOBAL_OPTION:
//      Only present when the plugin is built with ENABLE_TP_DEBUG_CAPS.
//      These are TP wrappers rather than the OV_CONFIG_DEBUG_* macros: the
//      stock ones key off ENABLE_DEBUG_CAPS, which would let the options exist
//      -- and accept an environment value -- in a build where the call sites
//      have been compiled out, so the value would be taken and then ignored.
//
// The environment variable name of an option is "OV_" + its property key, so
// `TP_INPUT_STAGE_MAX_BYTES` is set through `OV_TP_INPUT_STAGE_MAX_BYTES`.

// Namespace, property name, default value, [validator], description
OV_CONFIG_RELEASE_OPTION(ov::device, priorities, "", "Rank devices, one rank per entry in rank order; set through the device name, e.g. \"TP:GPU.0,GPU.1\"")
OV_CONFIG_RELEASE_OPTION(ov::tp, tp_strategy, ov::tp::TPStrategy::COLUMN_ROW, "Partitioning strategy. Values: COLUMN_ROW")
OV_CONFIG_RELEASE_OPTION(ov::tp, collective_backend, std::string{ov::tp::auto_backend}, "Collective implementation of the rank devices' plugin, passed to it unchanged; AUTO lets it choose")
OV_CONFIG_RELEASE_OPTION(ov::tp::internal, communication_timeout_ms, 5000, "Milliseconds a rank waits inside a collective before the whole group is aborted. 0 waits indefinitely", [](int v) { return v >= 0; })
OV_CONFIG_RELEASE_OPTION(ov::tp::internal, comm_buffer_size, size_t{0}, "Bytes of collective staging per rank, allocated once with the rank contexts; larger messages are split into chunks. 0 lets the staging grow with the largest message")

OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::tp, input_stage_max_bytes, uint64_t{4096}, "Size ceiling in bytes for staging a user input through plugin-owned host memory. 0 always stages on the device")
OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::tp, disable_lm_head_sharding, false, "Leave the vocabulary projection whole on every rank: less device memory when it shares its weight with the embedding, a slower decode step")

OV_CONFIG_TP_DEBUG_GLOBAL_OPTION(ov::tp, verbose, ov::log::Level::NO, "Verbosity of the plugin's diagnostics. Values: LOG_NONE, LOG_ERROR, LOG_WARNING, LOG_INFO, LOG_DEBUG, LOG_TRACE")
OV_CONFIG_TP_DEBUG_GLOBAL_OPTION(ov::tp, log_to_file, std::string{}, "File the plugin's diagnostics and measurement reports go to instead of the standard output")

OV_CONFIG_TP_DEBUG_OPTION(ov::tp, dump_sharding_states, std::string{}, "Directory where compile_model writes the sharding verifier's state of every split or annotated node, also when the verification fails")
OV_CONFIG_TP_DEBUG_OPTION(ov::tp, shard_only, false, "DANGEROUS: compile one rank's shard with every collective stripped. Output is wrong by construction")
