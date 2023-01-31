// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/icompiled_model.hpp"

#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "icompiled_model_wrapper.hpp"
#include "openvino/core/model.hpp"
#include "transformations/utils/utils.hpp"

ov::ICompiledModel::ICompiledModel(const std::shared_ptr<const ov::Model>& model,
                                   const std::shared_ptr<const ov::IPlugin>& plugin,
                                   const InferenceEngine::ITaskExecutor::Ptr& task_executor,
                                   const InferenceEngine::ITaskExecutor::Ptr& callback_executor)
    : m_plugin(plugin),
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

        if (!m_plugin->is_new_api()) {
            for (const auto& param : model->get_parameters()) {
                const auto& param_name = param->get_friendly_name();
                auto new_param = ov::as_type_ptr<ov::op::v0::Parameter>(param->copy_with_new_inputs({}));
                new_param->set_friendly_name(param_name);
                if (add_operation_names) {
                    OPENVINO_ASSERT(
                        !m_plugin->is_new_api() || leaf_names.find(param_name) == leaf_names.end() ||
                            param->output(0).get_names().find(param_name) != param->output(0).get_names().end(),
                        "Model operation names have collisions with tensor names.",
                        " Please use MO to generate new IR version, it should allow to avoid the issue");
                    leaf_names.insert(param_name);
                    new_param->output(0).get_tensor().add_names({param_name});
                }
                new_param->set_layout(param->get_layout());
                new_param->output(0).get_rt_info() = param->output(0).get_rt_info();
                new_param->validate_and_infer_types();
                _parameters.emplace_back(new_param);
                m_inputs.emplace_back(
                    ov::Output<const ov::Node>{new_param->output(0).get_node(), new_param->output(0).get_index()});
            }
            for (const auto& result : model->get_results()) {
                auto fake_param = std::make_shared<ov::op::v0::Parameter>(result->get_output_element_type(0),
                                                                          result->get_output_partial_shape(0));
                const std::string res_name = ov::op::util::create_ie_output_name(result->input_value(0));
                fake_param->set_friendly_name(res_name);
                fake_param->validate_and_infer_types();
                auto new_result = result->copy_with_new_inputs({fake_param});
                new_result->set_friendly_name(result->get_friendly_name());
                new_result->validate_and_infer_types();
                if (add_operation_names) {
                    OPENVINO_ASSERT(
                        !m_plugin->is_new_api() || leaf_names.find(res_name) == leaf_names.end() ||
                            result->output(0).get_names().find(res_name) != result->output(0).get_names().end(),
                        "Model operation names have collisions with tensor names.",
                        " Please use MO to generate new IR version, it should allow to avoid the issue");
                    leaf_names.insert(res_name);
                    new_result->output(0).get_tensor().add_names({res_name});
                }
                auto r = std::dynamic_pointer_cast<ov::op::v0::Result>(new_result);
                r->set_layout(result->get_layout());
                _results.emplace_back(new_result);
                auto out = r->input_value(0);
                m_outputs.emplace_back(ov::Output<const ov::Node>{out.get_node(), out.get_index()});
            }
        } else {
            m_inputs = model->inputs();
            m_outputs = model->outputs();
        }
    }
}

const std::vector<ov::Output<const ov::Node>>& ov::ICompiledModel::outputs() const {
    return m_outputs;
}

const std::vector<ov::Output<const ov::Node>>& ov::ICompiledModel::inputs() const {
    return m_inputs;
}
std::shared_ptr<InferenceEngine::IInferRequestInternal> ov::ICompiledModel::create_infer_request() const {
    return create_async_infer_request();
}

const std::shared_ptr<const ov::IPlugin>& ov::ICompiledModel::get_plugin() const {
    return m_plugin;
}
const InferenceEngine::ITaskExecutor::Ptr ov::ICompiledModel::get_task_executor() const {
    return m_task_executor;
}
const InferenceEngine::ITaskExecutor::Ptr ov::ICompiledModel::get_callback_executor() const {
    return m_callback_executor;
}

void ov::ICompiledModel::loaded_from_cache() {
    if (auto wrapper = dynamic_cast<InferenceEngine::ICompiledModelWrapper*>(this)) {
        wrapper->get_executable_network()->loadedFromCache();
        return;
    }
    OPENVINO_NOT_IMPLEMENTED;
}
