// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides CompiledModel class
 *
 * @file openvino/runtime/compiled_model.hpp
 */

#pragma once

#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/parameter.hpp"
#include "openvino/runtime/remote_context.hpp"

namespace InferenceEngine {
class IExecutableNetworkInternal;
}  // namespace InferenceEngine
namespace ov {
namespace runtime {

class Core;
class InferRequest;

/**
 * @brief This is an interface of an executable network
 */
class OPENVINO_RUNTIME_API CompiledModel {
    std::shared_ptr<void> _so;
    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> _impl;

    /**
     * @brief Constructs CompiledModel from the initialized std::shared_ptr
     * @param so Plugin to use. This is required to ensure that CompiledModel can work properly even if plugin
     * object is destroyed.
     * @param impl Initialized shared pointer
     */
    CompiledModel(const std::shared_ptr<void>& so,
                  const std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>& impl);
    friend class ov::runtime::Core;
    friend class ov::runtime::InferRequest;

public:
    /**
     * @brief A default constructor.
     */
    CompiledModel() = default;

    /**
     * @brief Get executable graph information from a device
     *
     * @return Model containing Executable Graph Info
     */
    std::shared_ptr<const Model> get_runtime_model() const;

    /**
     * @brief Get inputs of executable graph
     *
     * @return vector of inputs
     */
    std::vector<ov::Output<const ov::Node>> inputs() const;

    /**
     * @brief Get input of executable graph
     *
     * @return Model input or throw ov::Exception in case of several outputs
     */
    ov::Output<const ov::Node> input() const;

    /**
     * @brief Get input of executable graph
     *
     * @param i input index
     * @return Model input or throw ov::Exception if input wasn't found
     */
    ov::Output<const ov::Node> input(size_t i) const;

    /**
     * @brief Get input of executable graph
     *
     * @param tensor_name The input tensor name
     * @return Model output or throw ov::Exception if input wasn't found
     */
    ov::Output<const ov::Node> input(const std::string& tensor_name) const;

    /**
     * @brief Get outputs of executable graph model
     *
     * @return vector of outputs
     */
    std::vector<ov::Output<const ov::Node>> outputs() const;

    /**
     * @brief Get output of executable graph
     *
     * @return Model output or throw ov::Exception in case of several outputs
     */
    ov::Output<const ov::Node> output() const;

    /**
     * @brief Get output of executable graph
     *
     * @param i output index
     * @return Model output or throw ov::Exception if output wasn't found
     */
    ov::Output<const ov::Node> output(size_t i) const;

    /**
     * @brief Get output of executable graph
     *
     * @param tensor_name The output tensor name
     * @return Model output or throw ov::Exception if output wasn't found
     */
    ov::Output<const ov::Node> output(const std::string& tensor_name) const;

    /**
     * @brief Creates an inference request object used to infer the model.
     *
     * The created request has allocated input and output blobs (that can be changed later).
     *
     * @return InferRequest object
     */
    InferRequest create_infer_request();

    /**
     * @brief Exports the current executable network.
     *
     * @see Core::import_model
     *
     * @param model_stream Model output stream
     */
    void export_model(std::ostream& model_stream);

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
     * CompiledModel::get_metric with the SUPPORTED_CONFIG_KEYS key, but some of these keys cannot be changed
     * dynamically, e.g. DEVICE_ID cannot changed if an executable network has already been compiled for particular
     * device.
     *
     * @param name config key, can be found in ie_plugin_config.hpp
     * @return Configuration parameter value
     */
    Any get_config(const std::string& name) const;

    /**
     * @brief Gets general runtime metric for an executable network.
     *
     * It can be model name, actual device ID on
     * which executable network is running or all other properties which cannot be changed dynamically.
     *
     * @param name metric name to request
     * @return Metric parameter value
     */
    Any get_metric(const std::string& name) const;

    /**
     * @brief Returns pointer to plugin-specific shared context
     * on remote accelerator device that was used to create this CompiledModel
     * @return A context
     */
    RemoteContext get_context() const;

    /**
     * @brief Checks if current CompiledModel object is not initialized
     * @return true if current CompiledModel object is not initialized, false - otherwise
     */
    bool operator!() const noexcept;

    /**
     * @brief Checks if current CompiledModel object is initialized
     * @return true if current CompiledModel object is initialized, false - otherwise
     */
    explicit operator bool() const noexcept;
};

}  // namespace runtime
}  // namespace ov
