// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <ze_graph_ext.h>

#include <iostream>
#include <string>

#include "intel_npu/config/config.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "ze_graph_ext.h"

namespace intel_npu {

struct SerializedIR {
    std::shared_ptr<uint8_t> buffer = nullptr;
    size_t size = 0;
    std::optional<uint64_t> hash = std::nullopt;
};

/**
 * @brief Contain all required transformation on OpenVINO model in case for external compiler usage and
 *  providing forward compatibility (OV model with opset N+M, external compiler with opset N)
 */
namespace driver_compiler_utils {

/**
 * @brief Serializes the model using a format supported by the "VCL" interface.
 *
 * @param compilerVersion The compiler version reported by the driver.
 * @param supportedOpsetVersion The last operators set version supported by the compiler.
 * @param useBaseModelSerializer "true" means the legacy serializer will be used (weights will be copied), "false" means
 * the optimized one is used instead (weights pointers are stored instead).
 * @param computeModelHash If true, a hash of the model will also be returned.
 * @param storeWeightlessCacheAttribute If true, the returned serialized model will also contain within its runtime
 * information the WeightlessCacheAttributes stored using a custom format. This format can be interpreted by the
 * driver-compiler adapter in order to properly handle the "weights separation" feature.
 *
 * @returns The serialized model, along with its size and hash
 */
SerializedIR serializeIR(const std::shared_ptr<const ov::Model>& model,
                         ze_graph_compiler_version_info_t compilerVersion,
                         const uint32_t supportedOpsetVersion = 11,
                         const bool useBaseModelSerializer = true,
                         const bool computeModelHash = false,
                         const bool storeWeightlessCacheAttribute = false);

/**
 * @brief Serialize input / output information to string format.
 * @details Format:
 * --inputs_precisions="0:<input1Precision> [1:<input2Precision>]"
 * --inputs_layouts="0:<input1Layout> [1:<input2Layout>]"
 * --outputs_precisions="0:<output1Precision>"
 * --outputs_layouts="0:<output1Layout>"
 *
 * For older compiler versions, the name of the inputs/outputs may be used instead of their indices.
 *
 * Since the layout information is no longer an important part of the metadata values when using the 2.0 OV
 * API, the layout fields shall be filled with default values in order to assure the backward compatibility
 * with the driver.
 */
std::string serializeIOInfo(const std::shared_ptr<const ov::Model>& model, const bool useIndices);

std::string serializeConfig(const Config& config,
                            ze_graph_compiler_version_info_t compilerVersion,
                            bool turboSupported = false);

}  // namespace driver_compiler_utils
}  // namespace intel_npu
