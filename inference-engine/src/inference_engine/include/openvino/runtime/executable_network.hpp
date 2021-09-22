// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides ExecutableNetwork class
 *
 * @file openvino/runtime/executable_network.hpp
 */

#pragma once

#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "openvino/core/function.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/parameter.hpp"
#include "openvino/runtime/remote_context.hpp"

namespace InferenceEngine {
class IExecutableNetworkInternal;
class RemoteContext;
}  // namespace InferenceEngine
namespace ov {
namespace runtime {

class Core;

/**
 * @brief This is an interface of an executable network
 */
class INFERENCE_ENGINE_API_CLASS(ExecutableNetwork) {
    std::shared_ptr<void> _so;
    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> _impl;

    /**
     * @brief Constructs ExecutableNetwork from the initialized std::shared_ptr
     * @param so Plugin to use. This is required to ensure that ExecutableNetwork can work properly even if plugin
     * object is destroyed.
     * @param impl Initialized shared pointer
     */
    ExecutableNetwork(const std::shared_ptr<void>& so, const std::shared_ptr<ie::IExecutableNetworkInternal>& impl);
    friend class ov::runtime::Core;

public:
    /**
     * @brief A default constructor.
     */
    ExecutableNetwork() = default;

    /**
     * @brief Get executable graph information from a device
     *
     * @return Function containing Executable Graph Info
     */
    std::shared_ptr<const Function> get_runtime_function() const;

    /**
     * @brief Get parameters of executeble graph function
     *
     * @return vector of paramter nodes
     */
    ParameterVector get_parameters() const;

    /**
     * @brief Get results of executeble graph function
     *
     * @return vector of result nodes
     */
    ResultVector get_results() const;

    /**
     * @brief Creates an inference request object used to infer the network.
     *
     * The created request has allocated input and output blobs (that can be changed later).
     *
     * @return InferRequest object
     */
    InferRequest create_infer_request();

    /**
     * @brief Exports the current executable network.
     *
     * @see Core::ImportNetwork
     *
     * @param networkModel Network model output stream
     */
    void export_model(std::ostream& networkModel);

    /**
     * @brief Sets configuration for current executable network
     *
     * @param config Map of pairs: (config parameter name, config parameter value)
     */
    void set_config(const ParamMap& config);

    /** @brief Gets configuration for current executable network.
     *
     * The method is responsible to extract information
     * which affects executable network execution. The list of supported configuration values can be extracted via
     * ExecutableNetwork::get_metric with the SUPPORTED_CONFIG_KEYS key, but some of these keys cannot be changed
     * dynamically, e.g. DEVICE_ID cannot changed if an executable network has already been compiled for particular
     * device.
     *
     * @param name config key, can be found in ie_plugin_config.hpp
     * @return Configuration parameter value
     */
    Parameter get_config(const std::string& name) const;

    /**
     * @brief Gets general runtime metric for an executable network.
     *
     * It can be network name, actual device ID on
     * which executable network is running or all other properties which cannot be changed dynamically.
     *
     * @param name metric name to request
     * @return Metric parameter value
     */
    Parameter get_metric(const std::string& name) const;

    /**
     * @brief Returns pointer to plugin-specific shared context
     * on remote accelerator device that was used to create this ExecutableNetwork
     * @return A context
     */
    std::shared_ptr<ie::RemoteContext> get_context() const;

    /**
     * @brief Checks if current ExecutableNetwork object is not initialized
     * @return true if current ExecutableNetwork object is not initialized, false - otherwise
     */
    bool operator!() const noexcept;

    /**
     * @brief Checks if current ExecutableNetwork object is initialized
     * @return true if current ExecutableNetwork object is initialized, false - otherwise
     */
    explicit operator bool() const noexcept;
};

}  // namespace runtime
}  // namespace ov
