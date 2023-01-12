// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime CompiledModel interface
 * @file icompiled_model.hpp
 */

#pragma once

#include <memory>
#include <openvino/runtime/common.hpp>
#include <ostream>
#include <vector>

#include "cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "threading/ie_cpu_streams_executor.hpp"
#include "threading/ie_itask_executor.hpp"

namespace InferenceEngine {
class Core;
}

namespace ov {

class IPlugin;
class ICompiledModel;
class CompiledModelWrapper;
class ExecNetworkWrapper;

class OPENVINO_RUNTIME_API ICompiledModel : public std::enable_shared_from_this<ICompiledModel> {
public:
    ICompiledModel(const std::shared_ptr<ov::Model>& model,
                   const std::shared_ptr<const ov::IPlugin>& plugin,
                   const InferenceEngine::ITaskExecutor::Ptr& task_executor =
                       std::make_shared<InferenceEngine::CPUStreamsExecutor>(InferenceEngine::IStreamsExecutor::Config{
                           "Default"}),
                   const InferenceEngine::ITaskExecutor::Ptr& callback_executor =
                       std::make_shared<InferenceEngine::CPUStreamsExecutor>(InferenceEngine::IStreamsExecutor::Config{
                           "Callback"}));

    const std::vector<ov::Output<const ov::Node>>& outputs() const;
    const std::vector<ov::Output<const ov::Node>>& inputs() const;

    virtual std::shared_ptr<ov::IInferRequest> create_infer_request() const;

    virtual void export_model(std::ostream& model) const;

    virtual std::shared_ptr<ov::Model> get_runtime_model() const;

    virtual void set_property(const ov::AnyMap& properties);

    virtual ov::Any get_property(const std::string& name) const;

    virtual ov::RemoteContext get_context() const;

private:
    std::vector<ov::Output<const ov::Node>> m_inputs;
    std::vector<ov::Output<const ov::Node>> m_outputs;
    std::shared_ptr<const ov::IPlugin> m_plugin;
    bool m_loaded_from_cache = false;

    friend IPlugin;
    friend ExecNetworkWrapper;
    friend CompiledModelWrapper;
    friend InferenceEngine::Core;

protected:
    virtual std::shared_ptr<ov::IInferRequest> create_infer_request_impl() const;
    template <typename AsyncInferRequestType = ov::IAsyncInferRequest>
    std::shared_ptr<ov::IInferRequest> create_async_infer_request_from_sync() const {
        std::shared_ptr<ov::IInferRequest> syncRequestImpl = this->create_infer_request_impl();
        return std::make_shared<AsyncInferRequestType>(syncRequestImpl, m_task_executor, m_callback_executor);
    }

    InferenceEngine::ITaskExecutor::Ptr m_task_executor = nullptr;      //!< Holds a task executor
    InferenceEngine::ITaskExecutor::Ptr m_callback_executor = nullptr;  //!< Holds a callback executor

    std::shared_ptr<const ov::IPlugin> get_plugin() const;
};

}  // namespace ov
