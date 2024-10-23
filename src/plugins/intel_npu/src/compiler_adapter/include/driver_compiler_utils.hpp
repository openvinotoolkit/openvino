// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <type_traits>
#include <utility>

#include "intel_npu/config/config.hpp"

namespace intel_npu {

using SerializedIR = std::pair<size_t, std::shared_ptr<uint8_t>>;

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

SerializedIR serializeIR(const std::shared_ptr<const ov::Model>& model,
                         ze_graph_compiler_version_info_t compilerVersion,
                         const uint32_t supportedOpsetVersion);

std::string serializeConfig(const Config& config, ze_graph_compiler_version_info_t compilerVersion);

}  // namespace intel_npu
