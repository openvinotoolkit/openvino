// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/icompiled_model.hpp"

#include "openvino/core/model.hpp"

ov::ICompiledModel::ICompiledModel(const std::shared_ptr<const ov::Model>& model,
                                   const std::shared_ptr<const ov::IPlugin>& plugin,
                                   const InferenceEngine::ITaskExecutor::Ptr& task_executor,
                                   const InferenceEngine::ITaskExecutor::Ptr& callback_executor)
    : m_plugin(plugin),
      m_task_executor(task_executor),
      m_callback_executor(callback_executor) {
    if (model) {
        // Initialize inputs/outputs
        m_inputs = model->inputs();
        m_outputs = model->outputs();
    }
}

const std::vector<ov::Output<const ov::Node>>& ov::ICompiledModel::outputs() const {
    return m_outputs;
}

const std::vector<ov::Output<const ov::Node>>& ov::ICompiledModel::inputs() const {
    return m_inputs;
}
std::shared_ptr<InferenceEngine::IInferRequestInternal> ov::ICompiledModel::create_infer_request() const {
    if (m_task_executor && m_callback_executor) {
        return create_async_infer_request();
    }
    return create_sync_infer_request();
}

std::shared_ptr<const ov::IPlugin> ov::ICompiledModel::get_plugin() const {
    return m_plugin;
}
