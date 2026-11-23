// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>

#include "intel_gpu/runtime/execution_config.hpp"
#include "openvino/core/model.hpp"

namespace ov::intel_gpu::mlir {

// Replaces the supported subgraphs of the model with MLIROp and starts the background compilation of
// each of them. 'config' is needed for the per-model ov::intel_gpu::mlir_patterns option and
// 'device_id' is the hardware device ID the programs are compiled for.
void transformMLIR(const std::shared_ptr<ov::Model>& model, const ExecutionConfig& config, uint32_t device_id);

}  // namespace ov::intel_gpu::mlir
