// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/core/model.hpp"

namespace intel_npu {

/**
 * @brief Automatically selects the "HostCompile_Interpreter" compilation mode for fully-dynamic models.
 *
 * The mode is enabled only for Plugin compiler requests that have no explicit compilation mode and no
 * dynamic-to-static conversion, when both inputs and outputs contain a bounded dynamic static-rank 4D port (assumed
 * N,C,H,W order) and every I/O port dimension has a finite upper bound (HostCompile allocates dynamic buffers from
 * these upper bounds). Only height (H) and width (W) determine candidacy - the channel (C) dimension's static/dynamic
 * state is not considered, and a dynamic batch (N) alone does not matter either: "H", "W", "HW", "NH", "NW" and
 * "NHW" are all accepted, and a dynamic batch (N) with both H and W static ("N alone") is the only combination that
 * falls back to the regular batch handling path.
 *
 * @param model  Model being compiled.
 * @param config Configuration updated in place with the selected compilation mode when the criteria are met.
 * @param logger Logger used to report the automatic selection.
 * @return True if "HostCompile_Interpreter" was selected, false otherwise.
 */
bool enable_host_compile_if_needed(const std::shared_ptr<const ov::Model>& model,
                                   FilteredConfig& config,
                                   const Logger& logger = Logger::global());

/**
 * @brief Tells whether a model must be compiled through the HostCompile dynamic-graph path.
 *
 * HostCompile dynamic models keep their dynamic dimensions for the VM runtime instead of plugin-side debatching. This
 * is the case for a dynamic model compiled by the Plugin compiler when the resolved compilation mode starts with
 * "HostCompile" (either selected automatically by @ref enable_host_compile_if_needed or set explicitly by the user).
 *
 * @param model  Model being compiled.
 * @param config Configuration holding the resolved compiler type and compilation mode.
 * @return True if the HostCompile dynamic-graph path must be used, false otherwise.
 */
bool uses_host_compile_dynamic_graph(const std::shared_ptr<const ov::Model>& model, const Config& config);

}  // namespace intel_npu
