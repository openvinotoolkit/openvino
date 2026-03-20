// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lora_stateful_to_stateless.hpp"

#include "../util.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/util/node_util.hpp"
#include "openvino/op/util/assign_base.hpp"
#include "openvino/op/util/read_value_base.hpp"

namespace {

void convert_stateful_lora_to_stateless(const std::shared_ptr<ov::Model>& model) {
    typedef std::shared_ptr<ov::op::util::AssignBase> PAssign;
    typedef std::shared_ptr<ov::op::util::ReadValueBase> PReadValue;
    std::vector<PReadValue> readValues;
    std::vector<PAssign> assigns;
    auto sinks = model->get_sinks();
    for (size_t i = 0; i < sinks.size(); ++i) {
        if (auto assign = ov::as_type_ptr<ov::op::util::AssignBase>(sinks[i])) {
            auto variable_name = assign->get_variable_id();
            if (!ov::npuw::util::matchLoRAMatMulAString(variable_name) &&
                !ov::npuw::util::matchLoRAMatMulBString(variable_name) &&
                !ov::npuw::util::matchLoRAMatMulAlphaString(variable_name)) {
                continue;
            }

            auto read_value = ov::as_type_ptr<ov::op::util::ReadValueBase>(assign->get_input_node_shared_ptr(0));
            OPENVINO_ASSERT(read_value, "Can't find ReadValue");
            readValues.push_back(read_value);
            assigns.push_back(assign);
        }
    }

    ov::ParameterVector new_parameters;
    new_parameters.reserve(readValues.size());
    for (size_t i = 0; i < readValues.size(); ++i) {
        auto read_value = readValues[i];
        auto variable_name = read_value->get_variable_id();
        const auto element_type = read_value->get_output_element_type(0);
        const auto shape = read_value->get_output_partial_shape(0);

        auto parameter = std::make_shared<ov::op::v0::Parameter>(element_type, shape);
        ov::op::util::set_name(*parameter, variable_name);
        replace_node(read_value, parameter);

        auto assign = assigns[i];
        model->remove_sink(assign);
        model->remove_variable(model->get_variable_by_id(variable_name));
        new_parameters.push_back(parameter);
    }

    model->add_parameters(new_parameters);
}

}  // namespace

bool LoraStatefulToStatelessPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    convert_stateful_lora_to_stateless(model);

    return true;
}
