// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/replace_inputs_outputs_with_memory.h"

#include <memory>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/op/util/variable.hpp>
#include <openvino/opsets/opset8.hpp>

NGRAPH_RTTI_DEFINITION(ov::pass::MakeStateful, "MakeStateful", 0);

using namespace std;
using namespace ngraph;
using namespace opset8;
using namespace op::util;

namespace {
string generate_variable_name(const shared_ptr<Parameter>& param, const shared_ptr<Result>& res) {
    // todo variable name?
    return param->get_friendly_name() + res->get_friendly_name();
}
}  // namespace

bool ov::pass::MakeStateful::run_on_function(std::shared_ptr<ngraph::Function> f) {
    VariableVector variables;
    SinkVector sinks;
    for (const auto& pair : m_pairs_to_replace) {
        const auto& param = pair.first;
        const auto& res = pair.second;
        const auto& target_inputs = param->get_output_target_inputs(0);

        std::string var_name = generate_variable_name(param, res);
        auto variable = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, var_name});
        variables.push_back(variable);

        // create ReadValue
        /*        auto const_zero = make_shared<Constant>(param->get_element_type(), ngraph::Shape{1}, 0);
                auto shape_of = make_shared<ShapeOf>(param);

                auto broadcast = make_shared<Broadcast>(const_zero, shape_of);*/
        auto read_val = make_shared<ReadValue>(param, variable);
        for (const auto& target_in : target_inputs) {
            target_in.replace_source_output(read_val->output(0));
        }
        copy_runtime_info(param, read_val);

        // create Assign
        auto assign = make_shared<Assign>(res->input_value(0), variable);
        sinks.push_back(assign);
        copy_runtime_info(res, assign);

        assign->add_control_dependency(read_val);
    }
    f->add_variables(variables);
    f->add_sinks(sinks);
    return true;
}

ov::pass::MakeStateful::InOutPairs ov::pass::MakeStateful::findInputsOutputsByName(
    const shared_ptr<ngraph::Function>& func,
    const vector<std::pair<std::string, std::string>>& param_res_names) {
    InOutPairs pairs_to_replace;
    const auto& params = func->get_parameters();
    const auto& results = func->get_results();

    // find corresponding param and result by name and add to the list
    for (const auto& param_res : param_res_names) {
        const auto& param_name = param_res.first;
        const auto& res_name = param_res.second;

        auto param = std::find_if(params.begin(), params.end(), [&](const std::shared_ptr<ngraph::Node>& node) {
            return node->get_friendly_name().find(param_name) != std::string::npos;
        });
        NGRAPH_CHECK(param != params.end(),
                     "Parameter node with name = ",
                     param_name,
                     "doesn't belong to the function");

        auto res = std::find_if(results.begin(), results.end(), [&](const std::shared_ptr<ngraph::Node>& node) {
            return node->get_friendly_name().find(res_name) != std::string::npos;
        });
        NGRAPH_CHECK(res != results.end(), "Result node with name = ", res_name, " doesn't belong to the function");

        pairs_to_replace.emplace_back(*param, *res);
    }
    return pairs_to_replace;
}
