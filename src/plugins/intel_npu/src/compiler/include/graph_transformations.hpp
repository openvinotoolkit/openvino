// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "driver_compiler_adapter.hpp"
#include "openvino/pass/manager.hpp"

/**
 * @brief Contain all required transformation on OpenVINO model in case for external compiler usage and
 *  providing forward compatibility (OV model with opset N+M, external compiler with opset N)
 */
namespace intel_npu::driverCompilerAdapter {

/**
 * @brief Serialize OpenVINO model to IR
 */
IR serializeToIR(const std::shared_ptr<const ov::Model>& model, uint32_t supportedVersionByCompiler = 7);

}  // namespace intel_npu::driverCompilerAdapter
