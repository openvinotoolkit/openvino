// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <optional>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "ivcl_compiler.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/tensor.hpp"

namespace intel_npu {

class VCLCompilerImpl final : public IVCLCompiler, public std::enable_shared_from_this<VCLCompilerImpl> {
public:
    /**
     * @brief Builds a compiler on top of the given VCL entry points.
     * @param api The VCL dispatch table to call through. Injecting it is what makes this class
     *        testable: production code passes `VCLApi::getInstance(libraryDir)`, tests pass a table
     *        built with `VCLApi::NoLoad` and populated with fakes.
     */
    VCLCompilerImpl(std::shared_ptr<const VCLApi> api,
                    const std::optional<IDevice::DeviceProperties>& deviceProperties = std::nullopt);
    ~VCLCompilerImpl() override;

    /**
     * @brief Transforms a network from the OpenVINO model representation to a format executable
     * by a NPU device
     * @param model a shared pointer to the OpenVINO model to be compiled
     * @param config a reference to NPUConfig containing plugin config options
     *        including config options related to compilation
     * @return a pair containing an ov::Tensor object with the compiled model (blob) and an optional
     *         string with runtime requirements for the blob
     */
    std::pair<ov::Tensor, std::optional<std::string>> compile(const std::shared_ptr<const ov::Model>& model,
                                                              const FilteredConfig& config) const override;

    /**
     * @brief Compiles the model, weights separation enabled. All init schedules along with the main one are compiled in
     * the same scope.
     * @return A pair containing one ov::Tensor for each init schedule, followed by another one corresponding to the
     * main part, and an optional compatibility string for the compiled blobs.
     */
    std::pair<std::vector<ov::Tensor>, std::optional<std::string>> compileWsOneShot(
        const std::shared_ptr<ov::Model>& model,
        const FilteredConfig& config) const override;
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
     * Compiler should somehow understand which Init (or Main) to return
     * Plugin does not know total numbers of Init schedules
     */
    std::pair<ov::Tensor, std::optional<std::string>> compileWsIterative(const std::shared_ptr<ov::Model>& model,
                                                                         const FilteredConfig& config,
                                                                         size_t callNumber) const override;
    /**
     * @brief Returns information about supported layers of the network passed
     * @param model The model to be queried
     * @param config A reference to NPUConfig containing plugin config options
     *        including config options related to compilation
     * @returns SupportedOpsMap structure with information about supported layers
     */
    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model,
                              const FilteredConfig& config) const override;

    /**
     * @brief Returns the compiler version
     * @return composite uint32_t value of compiler version.
     *         MSB 16 bits = Major version
     *         LSB 16bits = Minor version
     */
    uint32_t get_version() const override;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const std::vector<uint8_t>& network) const override;

    /**
     * @brief Returns the compiler supported options list, NUL-trimmed and tokenised.
     */
    std::vector<std::string> get_supported_options() const override;

    /**
     * @brief Checks whether the given option and value are supported by the compiler
     * @param option The option name to check
     * @param optValue The option value to validate
     * @return true if the option and value are supported, false otherwise
     */
    bool is_option_supported(const std::string& option,
                             const std::optional<std::string>& optValue = std::nullopt) const override;

    std::shared_ptr<void> getLinkedLibrary() const;

private:
    /**
     * @brief Compiles the given model according to the given configuration. During the model serialization step,
     * the "WeightlessCacheAttribute" may be stored within the serialized model if requested.
     * @note Storing the "WeightlessCacheAttribute" is necessary if the "weights separation" flow is being used.
     */
    std::pair<ov::Tensor, std::optional<std::string>> compile(const std::shared_ptr<const ov::Model>& model,
                                                              const FilteredConfig& config,
                                                              const bool storeWeightlessCacheAttributeFlag) const;

    std::shared_ptr<const VCLApi> _api;
    vcl_log_handle_t _logHandle = nullptr;
    vcl_compiler_handle_t _compilerHandle = nullptr;
    vcl_compiler_properties_t _compilerProperties;
    vcl_version_info_t _vclVersion;
    vcl_version_info_t _vclProfilingVersion;
    Logger _logger;
};

/**
 * @brief Loads the VCL compiler library and returns a compiler paired with it.
 *
 * Keeps the load + SoPtr pairing in one place: the returned SoPtr owns the shared library, so the
 * compiler cannot outlive the code it dispatches into.
 */
ov::SoPtr<IVCLCompiler> makeVCLCompiler(
    const std::string& libraryDir,
    const std::optional<IDevice::DeviceProperties>& deviceProperties = std::nullopt);

}  // namespace intel_npu
