// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/icompiled_model.hpp"

#include <cpp/ie_executable_network.hpp>
#include <ie_remote_context.hpp>
#include <memory>

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "dev/converter_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/remote_context.hpp"

ov::ICompiledModel::ICompiledModel(const std::shared_ptr<ov::Model>& model,
                                   const std::shared_ptr<const ov::IPlugin>& plugin,
                                   const InferenceEngine::ITaskExecutor::Ptr& task_executor,
                                   const InferenceEngine::ITaskExecutor::Ptr& callback_executor)
    : m_plugin(plugin),
      m_task_executor(task_executor),
      m_callback_executor(callback_executor) {
    if (model) {
        std::shared_ptr<const ov::Model> const_model = model;
        // Add pre-processing
        m_inputs = const_model->inputs();
        m_outputs = const_model->outputs();
    }
}

const std::vector<ov::Output<const ov::Node>>& ov::ICompiledModel::outputs() const {
    return m_outputs;
}

const std::vector<ov::Output<const ov::Node>>& ov::ICompiledModel::inputs() const {
    return m_inputs;
}
std::shared_ptr<ov::IInferRequest> ov::ICompiledModel::create_infer_request() const {
    if (m_task_executor && m_callback_executor) {
        return create_async_infer_request_from_sync();
    }
    return create_infer_request_impl();
}

void ov::ICompiledModel::export_model(std::ostream& model) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::Model> ov::ICompiledModel::get_runtime_model() const {
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::ICompiledModel::set_property(const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::Any ov::ICompiledModel::get_property(const std::string& name) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::RemoteContext ov::ICompiledModel::get_context() const {
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::ICompiledModel::set_inputs(const std::vector<ov::Output<const ov::Node>>& inputs) {
    m_inputs = inputs;
}

void ov::ICompiledModel::set_outputs(const std::vector<ov::Output<const ov::Node>>& outputs) {
    m_outputs = outputs;
}

std::shared_ptr<ov::IInferRequest> ov::ICompiledModel::create_infer_request_impl() const {
    OPENVINO_NOT_IMPLEMENTED;
}
