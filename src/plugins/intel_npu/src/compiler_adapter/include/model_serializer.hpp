// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <ze_graph_ext.h>

#include <functional>
#include <iostream>
#include <string>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"

namespace intel_npu {

struct SerializedIR {
    std::shared_ptr<uint8_t> buffer = nullptr;
    size_t size = 0;
    ov::intel_npu::ModelSerializerVersion serializerVersion = MODEL_SERIALIZER_VERSION::defaultValue();
    std::optional<uint64_t> hash = std::nullopt;
};

/**
 * @brief Contain all required transformation on OpenVINO model and providing forward compatibility (OV model with opset
 * N+M, external compiler with opset N)
 */
namespace compiler_utils {

/**
 * @brief Serializes the model using a format supported by the "VCL" interface.
 *
 * @param compilerVersion The compiler version reported by the driver.
 * @param supportedOpsetVersion The last operators set version supported by the compiler.
 * @param serializerVersion The version of the serialization algorithm that should be applied. If not "AUTO", then only
 * the given version will be attempted. Otherwise, the NPU plugin will choose the version based on the support offered
 * by the compiler-adapter and preference.
 * @param isOptionSupportedByCompiler Function that allows querying the support offered by the compiler-adapter for a
 * given <config option, value> pair. The serializer will use this function to determine the compatibility of the
 * algorithm. If "nullptr" is passed, then the compatibility check is skipped.
 * @param computeModelHash If true, a hash of the model will also be returned.
 * @param storeWeightlessCacheAttribute If true, the returned serialized model will also contain within its runtime
 * information the WeightlessCacheAttributes stored using a custom format. This format can be interpreted by the
 * driver-compiler adapter in order to properly handle the "weights separation" feature.
 *
 * @returns The serialized model, along with its size and hash
 */
SerializedIR serializeIR(const std::shared_ptr<const ov::Model>& model,
                         ze_graph_compiler_version_info_t compilerVersion,
                         const uint32_t supportedOpsetVersion,
                         const ov::intel_npu::ModelSerializerVersion serializerVersion,
                         const std::function<bool(const std::string&, const std::optional<std::string>&)>&
                             isOptionSupportedByCompiler = nullptr,
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

std::string serializeConfig(const FilteredConfig& config,
                            const ze_graph_compiler_version_info_t& compilerVersion,
                            const std::function<bool(const std::string&)>& isOptionSupportedByCompiler);

/**
 * @brief Registers the passes that adapt a model to whichever compiler package is currently loaded, before it is
 * handed to that compiler.
 * @details The opset-downgrade and Identity-elimination passes are gated on the compiler's reported opset/version.
 * ov::pass::GroupQueryAttentionDecomposition is registered unconditionally instead: the compiler runs its own copy
 * of this pass, built against its own OpenVINO revision, and neither that revision nor a GQA-spec capability is
 * queryable at runtime - so a stale copy (wrong optional-input convention, or an attribute it predates, e.g.
 * local_window_size) can't be detected and worked around from here. A MatcherPass is a no-op where the operator is
 * absent, so this is safe to always register; delete it once the loaded compiler is known to be caught up.
 * @param manager The pass manager the compatibility passes are registered on; not run by this function.
 * @param supportedOpset The last operator set version supported by the compiler.
 * @param compilerVersion The compiler version reported by the driver.
 * @param logger Used to report which compatibility passes were registered and why.
 */
void registerCompilerCompatibilityPasses(ov::pass::Manager& manager,
                                         const uint32_t supportedOpset,
                                         const ze_graph_compiler_version_info_t& compilerVersion,
                                         Logger& logger);

}  // namespace compiler_utils
}  // namespace intel_npu
