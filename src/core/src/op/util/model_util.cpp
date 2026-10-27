// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/model_util.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/util/node_util.hpp"

namespace ov::util {
namespace {

void set_default_tensor_names(OutputVector&& outputs) {
    for (auto& output : outputs) {
        if (output.get_tensor().get_names().empty()) {
            output.get_tensor().set_names({make_default_tensor_name(output)});
        }
    }
}

void set_tensor_names(OutputVector&& outputs, const TensorNamesMap& tensor_names) {
    for (const auto& [port, names] : tensor_names) {
        if (port < outputs.size()) {
            outputs[port].get_tensor().set_names(names);
        }
    }
}
}  // namespace

void add_model_shared_objects(Model& model, const std::vector<std::shared_ptr<void>>& shared_objects) {
    if (shared_objects.empty())
        return;
    auto ownership = std::make_shared<std::vector<std::shared_ptr<void>>>(shared_objects);
    if (model.m_shared_object)
        ownership->push_back(model.m_shared_object);
    model.m_shared_object = std::move(ownership);
}

std::shared_ptr<Model> copy_model_with_shared_objects(const Model& model,
                                                      const std::vector<std::shared_ptr<void>>& shared_objects) {
    auto copy = std::make_shared<Model>(model.get_results(),
                                        model.get_sinks(),
                                        model.get_parameters(),
                                        model.get_variables(),
                                        model.get_friendly_name());
    copy->m_shared_object = model.m_shared_object;
    add_model_shared_objects(*copy, shared_objects);
    copy->get_rt_info() = model.get_rt_info();
    return copy;
}

void set_input_tensors_names(Model& model, const TensorNamesMap& inputs_names) {
    set_tensor_names(model.inputs(), inputs_names);
}

void set_input_tensors_names(const AutoTag&, Model& model, const TensorNamesMap& inputs_names) {
    set_input_tensors_names(model, inputs_names);
    set_default_tensor_names(model.inputs());
}

void set_output_tensor_names(Model& model, const TensorNamesMap& outputs_names) {
    set_tensor_names(model.outputs(), outputs_names);
}

void set_output_tensor_names(const AutoTag&, Model& model, const TensorNamesMap& outputs_names) {
    set_output_tensor_names(model, outputs_names);
    set_default_tensor_names(model.outputs());
}

void set_tensors_names(Model& model, const TensorNamesMap& inputs_names, const TensorNamesMap& outputs_names) {
    set_input_tensors_names(model, inputs_names);
    set_output_tensor_names(model, outputs_names);
}

void set_tensors_names(const AutoTag&,
                       Model& model,
                       const TensorNamesMap& inputs_names,
                       const TensorNamesMap& outputs_names) {
    set_input_tensors_names(AUTO, model, inputs_names);
    set_output_tensor_names(AUTO, model, outputs_names);
}

}  // namespace ov::util
