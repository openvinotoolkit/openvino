// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime CompiledModel interface
 * @file openvino/runtime/icompiled_model.hpp
 */

#pragma once

#include <memory>
#include <ostream>
#include <vector>

#include "cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "threading/ie_cpu_streams_executor.hpp"
#include "threading/ie_itask_executor.hpp"

namespace InferenceEngine {
class IInferRequestInternal;
class ICompiledModelWrapper;
}  // namespace InferenceEngine

namespace ov {

class CoreImpl;
class IPlugin;
class IExecutableNetworkWrapper;

/**
 * @brief OpenVINO ICompiledModel interface
 */
class OPENVINO_RUNTIME_API ICompiledModel : public std::enable_shared_from_this<ICompiledModel> {
public:
    /**
     * @brief Main constructor for ICompiledModel interface
     *
     * @param model OpenVINO model representation
     *
     * @param plugin Pointer to plugin
     *
     * @param task_executor Task executor (CPUStreamsExecutor by default)
     *
     * @param callback_executor Callback executor (CPUStreamsExecutor by default)
     */
    ICompiledModel(const std::shared_ptr<const ov::Model>& model,
                   const std::shared_ptr<const ov::IPlugin>& plugin,
                   const InferenceEngine::ITaskExecutor::Ptr& task_executor =
                       std::make_shared<InferenceEngine::CPUStreamsExecutor>(InferenceEngine::IStreamsExecutor::Config{
                           "Default"}),
                   const InferenceEngine::ITaskExecutor::Ptr& callback_executor =
                       std::make_shared<InferenceEngine::CPUStreamsExecutor>(InferenceEngine::IStreamsExecutor::Config{
                           "Callback"}));

    /**
     * @brief Gets all outputs from compiled model
     *
     * @return model outputs
     */
    const std::vector<ov::Output<const ov::Node>>& outputs() const;

    /**
     * @brief Gets all inputs from compiled model
     *
     * @return model inputs
     */
    const std::vector<ov::Output<const ov::Node>>& inputs() const;

    /**
     * @brief Create infer request
     *
     * @return Infer request interface
     */
    virtual std::shared_ptr<InferenceEngine::IInferRequestInternal> create_infer_request() const;

    /**
     * @brief Export compiled model to stream
     *
     * @param model output stream
     */
    virtual void export_model(std::ostream& model) const = 0;

    /**
     * @brief Returns runtime model
     *
     * @return OpenVINO Model which represents runtime graph
     */
    virtual std::shared_ptr<ov::Model> get_runtime_model() const = 0;

    /**
     * @brief Allows to set propertu
     *
     * @param properties new plugin properties
     */
    virtual void set_property(const ov::AnyMap& properties) = 0;

    /**
     * @brief Returns property
     *
     * @param name Property name
     *
     * @return Property value
     */
    virtual ov::Any get_property(const std::string& name) const = 0;

    /**
     * @brief Returns property requested from the specified device
     *
     * @param name Property name
     * @param target_device device name
     *
     * @return Property value
     */
    virtual ov::Any get_property(const std::string& name, const std::string& target_device) const = 0;

    /**
     * @brief Creates device specific remote context
     *
     * @return OpenVINO RemoteContext
     */
    virtual ov::RemoteContext get_context() const = 0;

private:
    std::shared_ptr<const ov::IPlugin> m_plugin;
    std::vector<ov::Output<const ov::Node>> m_inputs;
    std::vector<ov::Output<const ov::Node>> m_outputs;

    InferenceEngine::ITaskExecutor::Ptr m_task_executor = nullptr;      //!< Holds a task executor
    InferenceEngine::ITaskExecutor::Ptr m_callback_executor = nullptr;  //!< Holds a callback executor

    friend ov::CoreImpl;
    friend ov::IExecutableNetworkWrapper;
    friend InferenceEngine::ICompiledModelWrapper;

    /**
     * @brief function allows to mark that model was loaded from cache
     */
    void loaded_from_cache();

protected:
    /**
     * @brief Method creates infer request implementation
     *
     * @return Sync infer request
     */
    virtual std::shared_ptr<InferenceEngine::IInferRequestInternal> create_sync_infer_request() const = 0;

    /**
     * @brief Default imolementation of create async inter request method
     *
     * @tparam AsyncInferRequestType Async infer request type. InferenceEngine::AsyncInferRequestThreadSafeDefault by
     * default
     *
     * @return Async infer request
     */
    template <typename AsyncInferRequestType = InferenceEngine::AsyncInferRequestThreadSafeDefault>
    std::shared_ptr<InferenceEngine::IInferRequestInternal> create_async_infer_request() const {
        std::shared_ptr<InferenceEngine::IInferRequestInternal> syncRequestImpl = this->create_sync_infer_request();
        return std::make_shared<AsyncInferRequestType>(syncRequestImpl, m_task_executor, m_callback_executor);
    }

    /**
     * @brief Returns pointer to the plugin
     *
     * @return OpenVINO Plugin interface
     */
    std::shared_ptr<const ov::IPlugin> get_plugin() const;
};

}  // namespace ov
