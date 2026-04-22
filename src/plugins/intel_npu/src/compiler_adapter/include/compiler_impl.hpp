// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <optional>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/tensor.hpp"

namespace intel_npu {

class VCLCompilerImpl final : public std::enable_shared_from_this<VCLCompilerImpl> {
public:
    VCLCompilerImpl();
    ~VCLCompilerImpl();
    static const std::shared_ptr<VCLCompilerImpl> getInstance();

    /**
     * @brief Transforms a network from the OpenVINO model representation to a format executable
     * by a NPU device
     * @param model a shared pointer to the OpenVINO model to be compiled
     * @param config a reference to NPUConfig containing plugin config options
     *        including config options related to compilation
     * @return an ov::Tensor object containing the blob of the compiled model
     */
    std::pair<ov::Tensor, std::optional<std::string>> compile(const std::shared_ptr<const ov::Model>& model,
                                                              const FilteredConfig& config) const;

    /**
     * @brief Compiles the model, weights separation enabled. All init schedules along with the main one are compiled in
     * the same scope.
     * @return An ov::Tensor object for each init schedule, followed by another one corresponding to the main
     * part.
     */
    std::vector<ov::Tensor> compileWsOneShot(const std::shared_ptr<ov::Model>& model,
                                             const FilteredConfig& config) const;
    /**
     * @brief Sequential compilation of Init(s) and Main
     *
     * "Stateless compiler" approach
     * We want to get multiple Inits in the case of a large number of weights.
     * This allows us to build pipeline:
     * Allocate W1 -> Init1
     *             Allocate W2 -> Init2
     *                          Allocate W3 -> Init2
     *
     * This is why there is an additional parameter callNumber:
     * Compiler should somehow understand wich Init(or Main) to return
     * Plugin does not know total numbers of Init schedules
     */
    ov::Tensor compileWsIterative(const std::shared_ptr<ov::Model>& model,
                                  const FilteredConfig& config,
                                  size_t callNumber) const;
    /**
     * @brief Returns information about supported layers of the network passed
     * @param model The model to be queried
     * @param config A reference to NPUConfig containing plugin config options
     *        including config options related to compilation
     * @returns SupportedOpsMap structure with information about supported layers
     */
    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const FilteredConfig& config) const;

    /**
     * @brief Returns the compiler version
     * @return composite uint32_t value of compiler version.
     *         MSB 16 bits = Major version
     *         LSB 16bits = Minor version
     */
    uint32_t get_version() const;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const std::vector<uint8_t>& network) const;

    /**
     * @brief Returns the compiler supported options list
     * @return false if the API is not supported, true otherwise
     */
    bool get_supported_options(std::vector<char>& options) const;

    bool is_option_supported(std::string option, std::optional<std::string> optValue = std::nullopt) const;

    std::shared_ptr<void> getLinkedLibrary() const;

    ov::RuntimeRequirementCheckResult validate_compatibility_descriptor(const std::string& compatibilityDescriptor,
                                                                        uint32_t deviceId,
                                                                        int64_t numTiles,
                                                                        int64_t stepping) const;

private:
    /**
     * @brief Compiles the given model according to the given configuration. During the model serialization step,
     * the "WeightlessCacheAttribute" may be stored within the serialized model if requested.
     * @note Storing the "WeightlessCacheAttribute" is necessary if the "weights separation" flow is being used.
     */
    std::pair<ov::Tensor, std::optional<std::string>> compile(const std::shared_ptr<const ov::Model>& model,
                                                              const FilteredConfig& config,
                                                              const bool storeWeightlessCacheAttributeFlag) const;

    vcl_log_handle_t _logHandle = nullptr;
    vcl_compiler_handle_t _compilerHandle = nullptr;
    vcl_compiler_properties_t _compilerProperties;
    vcl_version_info_t _vclVersion;
    vcl_version_info_t _vclProfilingVersion;
    Logger _logger;
};

}  // namespace intel_npu
