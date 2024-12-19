// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include "intel_npu/config/config.hpp"
#include "intel_npu/network_metadata.hpp"
#include "openvino/runtime/profiling_info.hpp"

namespace intel_npu {

/**
 * @struct NetworkDescription
 * @brief The object returned by the compiler
 * to provide such information about a network as description of inputs and outputs,
 * name and compiled network in a format executable by device
 */
struct NetworkDescription final {
    NetworkDescription(std::vector<uint8_t>&& compiledNetwork, NetworkMetadata&& metadata)
        : compiledNetwork(std::move(compiledNetwork)),
          metadata(std::move(metadata)) {}
    // Force move semantics to prevent blob copies
    NetworkDescription(const NetworkDescription&) = delete;
    NetworkDescription(NetworkDescription&&) = default;
    NetworkDescription& operator=(const NetworkDescription&) = delete;
    NetworkDescription& operator=(NetworkDescription&&) = default;
    ~NetworkDescription() = default;

    std::vector<uint8_t> compiledNetwork;

    NetworkMetadata metadata;
};

/**
 * @interface ICompiler
 * @brief An interface to be implemented by a concrete compiler to provide
 * methods for preparing a network for execution on a NPU device
 */
class ICompiler : public std::enable_shared_from_this<ICompiler> {
public:
    /**
     * @brief Transforms a network from the OpenVINO model representation to a format executable
     * by a NPU device
     * @param model a shared pointer to the OpenVINO model to be compiled
     * @param config a reference to NPUConfig containing plugin config options
     *        including config options related to compilation
     * @return a shared pointer on an object implementing NetworkDescription interface
     */
    virtual NetworkDescription compile(const std::shared_ptr<const ov::Model>& model, const Config& config) const = 0;

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
     * @return a shared pointer on an object implementing NetworkDescription interface
     */
    virtual NetworkMetadata parse(const std::vector<uint8_t>& network, const Config& config) const = 0;

    virtual std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                                    const std::vector<uint8_t>& network,
                                                                    const Config& config) const = 0;

protected:
    virtual ~ICompiler() = default;
};

}  // namespace intel_npu
