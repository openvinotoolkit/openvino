// Copyright (C) 2018-2022 Intel Corporation
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
 * @brief This class represents compiled model
 * Model is compiled by a specific device by applying multiple optimization
 * transformations, then mapping to compute kernels.
 */
class OPENVINO_RUNTIME_API CompiledModel {
    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> _impl;
    std::shared_ptr<void> _so;

    /**
     * @brief Constructs CompiledModel from the initialized std::shared_ptr
     * @param impl Initialized shared pointer
     * @param so Plugin to use. This is required to ensure that CompiledModel can work properly even if plugin
     * object is destroyed.
     */
    CompiledModel(const std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>& impl,
                  const std::shared_ptr<void>& so);
    friend class ov::runtime::Core;
    friend class ov::runtime::InferRequest;

public:
    /**
     * @brief A default constructor.
     */
    CompiledModel() = default;

    /**
     * @brief Destructor preserves unloading order of implementation object and reference to library
     */
    ~CompiledModel();

    /**
     * @brief Get executable model information from a device
     * This object represents the internal device specific model which is optimized for particular
     * accelerator. It contains device specific nodes, runtime information and can be used only
     * to understand how the source model is optimized and which kernels, element types and layouts
     * are selected for optimal inference.
     *
     * @return Model containing Executable Graph Info
     */
    std::shared_ptr<const Model> get_runtime_model() const;

    /**
     * @brief Gets all inputs of a compiled model
     * Inputs are represented as a vector of outputs of ov::op::v0::Parameter operations.
     * They contain information about input tensors such as tensor shape, names and element type
     * @return std::vector of model inputs
     */
    std::vector<ov::Output<const ov::Node>> inputs() const;

    /**
     * @brief Gets a single input of a compiled model
     * An input is represented as an output of ov::op::v0::Parameter operation.
     * An input contain information about input tensor such as tensor shape, names and element type
     * @return A compiled model input
     * @note If a model has more than one input, this method throws an ov::Exception
     */
    ov::Output<const ov::Node> input() const;

    /**
     * @brief Gets input of a compiled model identified by an @p i
     * An input contains information about input tensor such as tensor shape, names and element type
     * @param i An input index
     * @return A compiled model input
     * @note The method throws ov::Exception if input with specified index @p i is not found
     */
    ov::Output<const ov::Node> input(size_t i) const;

    /**
     * @brief Gets input of a compiled model identified by a @p tensor_name
     * An input contain information about input tensor such as tensor shape, names and element type
     * @param tensor_name The input tensor name
     * @return A compiled model input
     * @note The method throws ov::Exception if input with specified tensor name @p tensor_name is not found
     */
    ov::Output<const ov::Node> input(const std::string& tensor_name) const;

    /**
     * @brief Get all outputs of a compiled model
     * Outputs are represented as a vector of output from ov::op::v0::Result operations.
     * Outputs contain information about output tensors such as tensor shape, names and element type
     * @return std::vector of model outputs
     */
    std::vector<ov::Output<const ov::Node>> outputs() const;

    /**
     * @brief Gets a single output of a compiled model
     * An output is represented as an output from ov::op::v0::Result operation.
     * An output contain information about output tensor such as tensor shape, names and element type
     * @return A compiled model output
     * @note If a model has more than one output, this method throws an ov::Exception
     */
    ov::Output<const ov::Node> output() const;

    /**
     * @brief Gets output of a compiled model identified by an @p index
     * An output contain information about output tensor such as tensor shape, names and element type
     * @param i An output index
     * @return A compiled model output
     * @note The method throws ov::Exception if output with specified index @p index is not found
     */
    ov::Output<const ov::Node> output(size_t i) const;

    /**
     * @brief Gets output of a compiled model identified by a @p tensor_name
     * An output contain information about output tensor such as tensor shape, names and element type
     * @param tensor_name The output tensor name
     * @return A compiled model output
     * @note The method throws ov::Exception if output with specified tensor name @p tensor_name is not found
     */
    ov::Output<const ov::Node> output(const std::string& tensor_name) const;

    /**
     * @brief Creates an inference request object used to infer the compiled model.
     * The created request has allocated input and output tensors (that can be changed later).
     *
     * @return InferRequest object
     */
    InferRequest create_infer_request();

    /**
     * @brief Exports the current compiled model to an output stream `std::ostream`.
     * The exported model can also be imported via ov::runtime::Core::import_model method
     * @see ov::runtime::Core::import_model
     * @param model_stream Output stream to store the model to
     */
    void export_model(std::ostream& model_stream);

    /**
     * @brief Sets configuration for current compiled model
     * @param config Map of pairs: (config parameter name, config parameter value)
     */
    void set_config(const ParamMap& config);

    /** @brief Gets configuration for a compiled model.
     *
     * The method is responsible to extract information
     * which affects compiled model inference. The list of supported configuration values can be extracted via
     * CompiledModel::get_metric with the SUPPORTED_CONFIG_KEYS key, but some of these keys cannot be changed
     * dynamically, e.g. DEVICE_ID cannot changed if a compiled model has already been compiled for particular
     * device.
     *
     * @param key_name config key, can be found in ie_plugin_config.hpp
     * @return Configuration parameter value
     */
    Any get_config(const std::string& key_name) const;

    /**
     * @brief Gets general runtime metric for a compiled model.
     *
     * It can be model name, actual device ID on
     * which compiled model is running or all other properties which cannot be changed dynamically.
     *
     * @param metric_name metric name to request
     * @return Metric parameter value
     */
    Any get_metric(const std::string& metric_name) const;

    /**
     * @brief Returns pointer to device-specific shared context
     * on remote accelerator device that was used to create this CompiledModel
     * @return A context
     */
    RemoteContext get_context() const;

    /**
     * @brief Checks if current CompiledModel object is not initialized
     * @return `true` if current CompiledModel object is not initialized, `false` - otherwise
     */
    bool operator!() const noexcept;

    /**
     * @brief Checks if current CompiledModel object is initialized
     * @return `true` if current CompiledModel object is initialized, `false` - otherwise
     */
    explicit operator bool() const noexcept;
};

}  // namespace runtime
}  // namespace ov
