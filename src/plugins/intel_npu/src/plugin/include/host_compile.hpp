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
 * dynamic-to-static conversion, when both inputs and outputs contain a bounded dynamic static-rank 4D port with a
 * static batch dimension and every I/O port dimension has a finite upper bound (HostCompile allocates dynamic buffers
 * from these upper bounds). Dynamic batch is intentionally excluded because the compiler's ConvertBatchedLayerTo1N and
 * AdjustScaleShiftForDWConv passes do not support dynamic reshape; such models use the regular batch handling path.
 *
 * @param model  Model being compiled.
 * @param config Configuration updated in place with the selected compilation mode when the criteria are met.
 * @param logger Logger used to report the automatic selection.
 * @return True if "HostCompile_Interpreter" was selected, false otherwise.
 */
bool enable_host_compile_if_needed(const std::shared_ptr<const ov::Model>& model,
                                   FilteredConfig& config,
                                   const Logger& logger = Logger::global());

}  // namespace intel_npu
