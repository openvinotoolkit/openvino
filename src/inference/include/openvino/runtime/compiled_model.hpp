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
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/remote_context.hpp"

namespace InferenceEngine {
class IExecutableNetworkInternal;
}  // namespace InferenceEngine

namespace ov {

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
    friend class ov::Core;
    friend class ov::InferRequest;

    void get_property(const std::string& name, ov::Any& to) const;

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
     * @brief Get runtime model information from a device
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
     * The exported model can also be imported via ov::Core::import_model method
     * @see ov::Core::import_model
     * @param model_stream Output stream to store the model to
     */
    void export_model(std::ostream& model_stream);

    /**
     * @brief Sets properties for current compiled model
     *
     * @param properties Map of pairs: (property name, property value)
     */
    void set_property(const AnyMap& properties);

    /**
     * @brief Sets properties for current compiled model
     *
     * @tparam Properties Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param properties Optional pack of pairs: (property name, property value)
     * @return nothing
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> set_property(Properties&&... properties) {
        set_property(AnyMap{std::forward<Properties>(properties)...});
    }

    /** @brief Gets properties for current compiled model
     *
     * The method is responsible to extract information
     * which affects compiled model inference. The list of supported configuration values can be extracted via
     * CompiledModel::get_property with the ov::supported_properties key, but some of these keys cannot be changed
     * dynamically, e.g. ov::device::id cannot changed if a compiled model has already been compiled for particular
     * device.
     *
     * @param name property key, can be found in openvino/runtime/properties.hpp
     * @return Property value
     */
    Any get_property(const std::string& name) const;

    /**
     * @brief Gets properties dedicated to device behaviour.
     *
     * The method is targeted to extract information which can be set via set_property method.
     *
     * @tparam T - type of returned value
     * @param property  - property  object.
     * @return Value of property.
     */
    template <typename T, PropertyMutability mutability>
    T get_property(const ov::Property<T, mutability>& property) const {
        auto to = Any::make<T>();
        get_property(property.name(), to);
        return to.template as<T>();
    }

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

namespace runtime {
using ov::CompiledModel;
}  // namespace runtime

}  // namespace ov
