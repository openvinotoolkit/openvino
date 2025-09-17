// Copyright (C) 2018-2025 Intel Corporation
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
