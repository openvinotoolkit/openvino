// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/icompiled_model.hpp"

#include "openvino/core/model.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/properties.hpp"
#include "transformations/utils/utils.hpp"

ov::ICompiledModel::ICompiledModel(const std::shared_ptr<const ov::Model>& model,
                                   const std::shared_ptr<const ov::IPlugin>& plugin,
                                   const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                                   const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : ICompiledModel(model, plugin, {}, task_executor, callback_executor) {}

ov::ICompiledModel::ICompiledModel(const std::shared_ptr<const ov::Model>& model,
                                   const std::shared_ptr<const ov::IPlugin>& plugin,
                                   const ov::SoPtr<ov::IRemoteContext>& context,
                                   const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                                   const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : m_plugin(plugin),
      m_context(context),
      m_task_executor(task_executor),
      m_callback_executor(callback_executor) {
    OPENVINO_ASSERT(m_plugin);
    if (model) {
        // Initialize inputs/outputs
        std::unordered_set<std::string> leaf_names;
        bool add_operation_names = false;
        if (model->has_rt_info("version")) {
            const int64_t ir_version = model->get_rt_info<int64_t>("version");
            // here we decide whether we need to add operation_names as tensor names for
            // getInputs / getOutputs. Since these functions are designed to be used in new API only
            // always need to add operation names for IR v10
            add_operation_names = ir_version == 10;

            if (add_operation_names) {
                for (const auto& vals : {model->inputs(), model->outputs()}) {
                    for (const auto& val : vals) {
                        for (const auto& name : val.get_names()) {
                            leaf_names.insert(name);
                        }
                    }
                }
            }
        }

        std::unordered_map<std::shared_ptr<ov::descriptor::Tensor>, std::shared_ptr<ov::descriptor::Tensor>> tensor_map;
        for (const auto& param : model->get_parameters()) {
            const auto& param_name = param->get_friendly_name();
            auto new_param = ov::as_type_ptr<ov::op::v0::Parameter>(param->copy_with_new_inputs({}));
            new_param->set_friendly_name(param_name);
            if (add_operation_names) {
                OPENVINO_ASSERT(leaf_names.find(param_name) == leaf_names.end() ||
                                    param->output(0).get_names().find(param_name) != param->output(0).get_names().end(),
                                "Model operation names have collisions with tensor names.",
                                " Please use MO to generate new IR version, it should allow to avoid the issue");
                leaf_names.insert(param_name);
                param->output(0).get_tensor().add_names({param_name});
                new_param->output(0).get_tensor().add_names({param_name});
            }
            new_param->set_element_type(param->get_element_type());
            new_param->set_layout(param->get_layout());
            new_param->output(0).get_rt_info() = param->output(0).get_rt_info();
            auto old_tensor = param->output(0).get_tensor_ptr();
            if (tensor_map.count(old_tensor)) {
                new_param->output(0).set_tensor_ptr(tensor_map[old_tensor]);
            } else {
                tensor_map[old_tensor] = new_param->output(0).get_tensor_ptr();
            }
            new_param->validate_and_infer_types();
            m_inputs.emplace_back(new_param->output(0));
        }
        for (const auto& result : model->get_results()) {
            auto fake_param = std::make_shared<ov::op::v0::Parameter>(result->get_output_element_type(0),
                                                                      result->get_output_partial_shape(0));
            OPENVINO_SUPPRESS_DEPRECATED_START
            const std::string res_name = ov::op::util::create_ie_output_name(result->input_value(0));
            OPENVINO_SUPPRESS_DEPRECATED_END
            fake_param->set_friendly_name(res_name);
            fake_param->set_element_type(result->get_element_type());
            fake_param->validate_and_infer_types();
            auto new_result = result->copy_with_new_inputs({fake_param});
            new_result->set_friendly_name(result->get_friendly_name());
            if (add_operation_names) {
                OPENVINO_ASSERT(leaf_names.find(res_name) == leaf_names.end() ||
                                    result->output(0).get_names().find(res_name) != result->output(0).get_names().end(),
                                "Model operation names have collisions with tensor names.",
                                " Please use MO to generate new IR version, it should allow to avoid the issue");
                leaf_names.insert(res_name);
                result->output(0).get_tensor().add_names({res_name});
                new_result->output(0).get_tensor().add_names({res_name});
            }
            auto r = std::dynamic_pointer_cast<ov::op::v0::Result>(new_result);
            r->set_layout(result->get_layout());
            new_result->output(0).get_rt_info() = result->output(0).get_rt_info();
            auto old_tensor = result->output(0).get_tensor_ptr();
            if (tensor_map.count(old_tensor)) {
                new_result->output(0).set_tensor_ptr(tensor_map[old_tensor]);
            } else {
                tensor_map[old_tensor] = new_result->output(0).get_tensor_ptr();
            }
            m_outputs.emplace_back(new_result->output(0));
        }
    }
}

const std::vector<ov::Output<const ov::Node>>& ov::ICompiledModel::outputs() const {
    return m_outputs;
}

const std::vector<ov::Output<const ov::Node>>& ov::ICompiledModel::inputs() const {
    return m_inputs;
}

std::shared_ptr<ov::IAsyncInferRequest> ov::ICompiledModel::create_infer_request() const {
    return create_async_infer_request();
}

const std::shared_ptr<const ov::IPlugin>& ov::ICompiledModel::get_plugin() const {
    return m_plugin;
}
const std::shared_ptr<ov::threading::ITaskExecutor> ov::ICompiledModel::get_task_executor() const {
    return m_task_executor;
}
const std::shared_ptr<ov::threading::ITaskExecutor> ov::ICompiledModel::get_callback_executor() const {
    return m_callback_executor;
}

void ov::ICompiledModel::set_task_executor(const std::shared_ptr<ov::threading::ITaskExecutor> task_executor) {
    m_task_executor = task_executor;
}

void ov::ICompiledModel::set_callback_executor(const std::shared_ptr<ov::threading::ITaskExecutor> callback_executor) {
    m_callback_executor = callback_executor;
}

ov::SoPtr<ov::IRemoteContext> ov::ICompiledModel::get_context() const {
    if (m_context)
        return m_context;
    return m_plugin->get_default_context({});
}

void ov::ICompiledModel::set_model_shared_object(ov::Model& model, const std::shared_ptr<void>& shared_object) {
    model.m_shared_object = shared_object;
}

void ov::ICompiledModel::release_memory() {
    // nothing to do
}
