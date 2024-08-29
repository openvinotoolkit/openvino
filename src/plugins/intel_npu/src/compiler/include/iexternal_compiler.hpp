// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <utility>

#include "intel_npu/al/icompiler.hpp"

namespace intel_npu {
namespace driverCompilerAdapter {

struct IR {
    std::stringstream xml;
    std::stringstream weights;
};

/**
 * @brief Interface for external compiler
 * @details Isolate external API calls from general logic
 */
class IExternalCompiler {
public:
    virtual ~IExternalCompiler() = default;

    /**
     * @brief Returns the maximum OpenVino opset version supported by the compiler
     * @return opset version e.g. 11 for opset11
     */
    virtual uint32_t getSupportedOpsetVersion() const = 0;

    /**
     * @brief Transforms a network from the OpenVINO model representation to a format executable
     * by a NPU device
     * @param model a shared pointer to the OpenVINO model to be compiled
     * @param config a reference to NPUConfig containing plugin config options
     *        including config options related to compilation
     * @return a shared pointer on an object implementing NetworkDescription interface
     */
    virtual std::pair<NetworkDescription, void*> compile(const std::shared_ptr<const ov::Model>& model,
                                                         const Config& config) const = 0;

    /**
     * @brief Returns information about supported layers of the network passed
     * @param model The model to be queried
     * @param config A reference to NPUConfig containing plugin config options
     *        including config options related to compilation
     * @returns SupportedOpsMap structure with information about supported layers
     */
    virtual ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const = 0;

    /**
     * @brief Parses already compiled network to extract meta information:
     *        inputs and outputs descriptions
     * @param network compiled network represented as a vector of char
     * @param config a reference to NPUConfig containing plugin config options
     *        Note: compilation options will be ignored,
     *        since the network is already compiled
     * @param netName a reference to the string describing network name
     *        to be used for creating network description
     * @return a shared pointer on an object implementing NetworkDescription interface
     */
    virtual std::pair<NetworkMetadata, void*> parse(const std::vector<uint8_t>& network,
                                                    const Config& config) const = 0;

    virtual std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                                    const std::vector<uint8_t>& network,
                                                                    const Config& config) const = 0;

    virtual void releaseGraphHandle(void* graphHandle) = 0;

    virtual void getCompiledNetwork(void* graphHandle, std::vector<uint8_t>& compiledNetwork) = 0;
};

}  // namespace driverCompilerAdapter
}  // namespace intel_npu