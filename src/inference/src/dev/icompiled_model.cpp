// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/icompiled_model.hpp"

#include "dev/converter_utils.hpp"
#include "icompiled_model_wrapper.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/properties.hpp"
#include "transformations/utils/utils.hpp"

ov::ICompiledModel::ICompiledModel(const std::shared_ptr<const ov::Model>& model,
                                   const std::shared_ptr<const ov::IPlugin>& plugin,
                                   const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                                   const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : ICompiledModel(model, plugin, {}, task_executor, callback_executor) {}

ov::ICompiledModel::ICompiledModel(const std::shared_ptr<const ov::Model>& model,
                                   const std::shared_ptr<const ov::IPlugin>& plugin,
                                   const ov::RemoteContext& context,
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

        if (add_operation_names) {
            for (const auto& param : model->get_parameters()) {
                const auto& param_name = param->get_friendly_name();
                OPENVINO_ASSERT(!m_plugin->is_new_api() || leaf_names.find(param_name) == leaf_names.end() ||
                                    param->output(0).get_names().find(param_name) != param->output(0).get_names().end(),
                                "Model operation names have collisions with tensor names.",
                                " Please use MO to generate new IR version, it should allow to avoid the issue");
                leaf_names.insert(param_name);
                param->output(0).get_tensor().add_names({param_name});
                m_inputs.emplace_back(
                    ov::Output<const ov::Node>{param->output(0).get_node(), param->output(0).get_index()});
            }
            for (const auto& result : model->get_results()) {
                auto fake_param = std::make_shared<ov::op::v0::Parameter>(result->get_output_element_type(0),
                                                                          result->get_output_partial_shape(0));
                const std::string res_name = ov::op::util::create_ie_output_name(result->input_value(0));
                OPENVINO_ASSERT(!m_plugin->is_new_api() || leaf_names.find(res_name) == leaf_names.end() ||
                                    result->output(0).get_names().find(res_name) != result->output(0).get_names().end(),
                                "Model operation names have collisions with tensor names.",
                                " Please use MO to generate new IR version, it should allow to avoid the issue");
                leaf_names.insert(res_name);
                result->output(0).get_tensor().add_names({res_name});
                m_outputs.emplace_back(
                    ov::Output<const ov::Node>{result->output(0).get_node(), result->output(0).get_index()});
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

std::shared_ptr<ov::IRemoteContext> ov::ICompiledModel::get_context() const {
    if (auto wrapper = dynamic_cast<const InferenceEngine::ICompiledModelWrapper*>(this)) {
        return ov::legacy_convert::convert_remote_context(wrapper->get_executable_network()->GetContext());
    }
    if (m_context._impl)
        return m_context._impl;
    return m_plugin->get_default_context({});
}

class FakeVariadicNode : public ov::op::Op {
public:
    OPENVINO_OP("FakeVariadicNode");

    FakeVariadicNode(const std::vector<ov::Output<const ov::Node>>& inputs,
                     const std::vector<ov::Output<const ov::Node>>& outputs)
        : ov::op::Op() {
        m_params.reserve(inputs.size());
        m_results.reserve(outputs.size());

        for (size_t i = 0; i < inputs.size(); i++) {
            const auto& input = inputs.at(i);
            auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(input.get_node()->clone_with_new_inputs({}));
            OPENVINO_ASSERT(param);
            param->set_friendly_name(input.get_node()->get_friendly_name());
            param->output(0).get_rt_info() = input.get_rt_info();
            param->validate_and_infer_types();
            m_params.emplace_back(param);
            set_argument(i, param);
        }
        set_output_size(outputs.size());
        for (size_t i = 0; i < outputs.size(); i++) {
            const auto& output = outputs.at(i);
            set_output_type(i, output.get_element_type(), output.get_partial_shape());
            const std::string res_name = ov::op::util::create_ie_output_name(output);
            OPENVINO_SUPPRESS_DEPRECATED_START
            ov::descriptor::set_ov_tensor_legacy_name(get_output_tensor(i), res_name);
            OPENVINO_SUPPRESS_DEPRECATED_END
            get_output_tensor(i).set_names(output.get_names());
        }
    }

    void validate_and_infer_types() override {}

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        // Cannot clone this fake operation
        OPENVINO_NOT_IMPLEMENTED;
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    const ov::ParameterVector& get_parameters() {
        return m_params;
    }

    const ov::ResultVector& get_results() {
        // Cannot init results in the constructor because shared_from_this will be called in make_shared line
        std::call_once(result_initialized, [this]() {
            for (const auto& output : outputs()) {
                auto result = std::make_shared<ov::op::v0::Result>(output);
                m_results.emplace_back(result);
            }
        });
        return m_results;
    }

private:
    std::vector<std::shared_ptr<ov::op::v0::Parameter>> m_params;
    std::vector<std::shared_ptr<ov::op::v0::Result>> m_results;
    std::once_flag result_initialized;
};

std::shared_ptr<ov::Model> ov::construct_model_with_inputs_outputs(
    const std::vector<ov::Output<const ov::Node>>& inputs,
    const std::vector<ov::Output<const ov::Node>>& outputs) {
    std::shared_ptr<FakeVariadicNode> fake_op(new FakeVariadicNode(inputs, outputs));
    return std::make_shared<ov::Model>(fake_op->get_results(), fake_op->get_parameters());
}
