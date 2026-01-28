// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <optional>

#include "compiler.h"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/network_metadata.hpp"
#include "openvino/core/except.hpp"

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
     * @return a shared pointer on an object implementing NetworkDescription interface
     */
    NetworkDescription compile(const std::shared_ptr<const ov::Model>& model, const Config& config) const;

    /**
     * @brief Compiles the model, weights separation enabled. All init schedules along with the main one are compiled in
     * the same scope.
     * @return A "NetworkDescription" object for each init schedule, followed by another one corresponding to the main
     * part.
     */
    std::vector<std::shared_ptr<NetworkDescription>> compileWsOneShot(const std::shared_ptr<ov::Model>& model,
                                                                      const Config& config) const;
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
    NetworkDescription compileWsIterative(const std::shared_ptr<ov::Model>& model,
                                          const Config& config,
                                          size_t callNumber) const;
    /**
     * @brief Returns information about supported layers of the network passed
     * @param model The model to be queried
     * @param config A reference to NPUConfig containing plugin config options
     *        including config options related to compilation
     * @returns SupportedOpsMap structure with information about supported layers
     */
    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const;

    /**
     * @brief Parses already compiled network to extract meta information:
     *        inputs and outputs descriptions
     * @param network compiled network represented as a vector of char
     * @param config a reference to NPUConfig containing plugin config options
     *        Note: compilation options will be ignored,
     *        since the network is already compiled
     * @return a shared pointer on an object implementing NetworkDescription interface
     */
    NetworkMetadata parse(const std::vector<uint8_t>& network, const Config& config) const;

    /**
     * @brief Returns the compiler version
     * @return composite uint32_t value of compiler version.
     *         MSB 16 bits = Major version
     *         LSB 16bits = Minor version
     */
    uint32_t get_version() const;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const std::vector<uint8_t>& network,
                                                            const intel_npu::Config& config) const;

    bool get_supported_options(std::vector<char>& options) const;

    bool is_option_supported(const std::string& option, std::optional<std::string> optValue = std::nullopt) const;

    std::shared_ptr<void> getLinkedLibrary() const;

private:
    /**
     * @brief Compiles the given model according to the given configuration. During the model serialization step, the
     * "WeightlessCacheAttribute" may be stored within the serialized model if requested.
     * @note Storing the "WeightlessCacheAttribute" is necessary if the "weights separation" flow is being used.
     */
    NetworkDescription compile(const std::shared_ptr<const ov::Model>& model,
                               const Config& config,
                               const bool storeWeightlessCacheAttributeFlag) const;

    vcl_log_handle_t _logHandle = nullptr;
    vcl_compiler_handle_t _compilerHandle = nullptr;
    vcl_compiler_properties_t _compilerProperties;
    vcl_version_info_t _vclVersion;
    vcl_version_info_t _vclProfilingVersion;
    Logger _logger;
};

}  // namespace intel_npu
